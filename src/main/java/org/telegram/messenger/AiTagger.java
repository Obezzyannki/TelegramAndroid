package org.telegram.messenger;

import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.text.TextUtils;
import android.util.Base64;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Auto-tagging for media captions via an OpenAI-compatible vision endpoint.
 * Everything is off unless the dialog id is in the whitelist.
 */
public class AiTagger {

    public static final String PREF_CHATS = "aiTagChats";
    public static final String PREF_BASE_URL = "aiTagBaseUrl";
    public static final String PREF_KEY = "aiTagKey";
    public static final String PREF_MODEL = "aiTagModel";
    public static final String PREF_PROMPT = "aiTagPrompt";
    public static final String PREF_FOOTER = "aiTagFooter";
    public static final String PREF_MAX_TAGS = "aiTagMaxTags";
    public static final String PREF_TEMPLATE = "aiTagTemplate";
    public static final String PREF_LINKS = "aiTagLinks";

    public static final String DEFAULT_MODEL = "deepseek-v4-flash-vision-exp";
    public static final String DEFAULT_PROMPT = "You tag images for a Telegram channel. Pick the most relevant subject, style and theme tags.";
    public static final int DEFAULT_MAX_TAGS = 12;
    public static final String DEFAULT_TEMPLATE = "( #char & #char ) & #char & #char / #generic_tags #generic_tags #body_part #body_part - #copyright [#additional]";

    private static final Pattern TAG = Pattern.compile("#[\\p{L}\\p{N}_]+");
    private static final ExecutorService executor = Executors.newCachedThreadPool();

    public interface Callback {
        /** Sanitized text so far (UI thread). */
        void onDelta(String tags);
        /** Final sanitized text (UI thread). */
        void onDone(String tags);
        /** One short message (UI thread). */
        void onError(String message);
    }

    public static class Job {
        volatile boolean cancelled;
        volatile HttpURLConnection conn;

        public void cancel() {
            cancelled = true;
            final HttpURLConnection c = conn;
            if (c != null) {
                try {
                    c.disconnect();
                } catch (Throwable ignore) {}
            }
        }
    }

    private static SharedPreferences prefs() {
        return MessagesController.getGlobalMainSettings();
    }

    // ---- settings ----

    /** Accepts bot-api style ids (-100123...) and raw telegram ids, comma/space/newline separated. */
    public static long normalizeId(long v) {
        if (v < -1000000000000L) {
            return -(-v - 1000000000000L);
        }
        return v;
    }

    public static Set<Long> getChats() {
        final Set<Long> out = new LinkedHashSet<>();
        final String raw = prefs().getString(PREF_CHATS, "");
        if (TextUtils.isEmpty(raw)) {
            return out;
        }
        for (String part : raw.split("[,;\\s]+")) {
            if (part.isEmpty()) continue;
            try {
                out.add(normalizeId(Long.parseLong(part)));
            } catch (NumberFormatException ignore) {}
        }
        return out;
    }

    public static String getChatsRaw() {
        return prefs().getString(PREF_CHATS, "");
    }

    public static boolean isWhitelisted(long dialogId) {
        if (dialogId == 0) {
            return false;
        }
        final String raw = prefs().getString(PREF_CHATS, "");
        if (TextUtils.isEmpty(raw)) {
            return false;
        }
        return getChats().contains(dialogId);
    }

    public static boolean isConfigured() {
        return !TextUtils.isEmpty(prefs().getString(PREF_BASE_URL, "")) && !TextUtils.isEmpty(prefs().getString(PREF_KEY, ""));
    }

    public static String getFooter() {
        return prefs().getString(PREF_FOOTER, "");
    }

    public static int getMaxTags() {
        return Math.max(1, Math.min(30, prefs().getInt(PREF_MAX_TAGS, DEFAULT_MAX_TAGS)));
    }

    /** Appends the footer (with [text](url) links as real entities) to the outgoing caption. Idempotent. */
    public static void appendFooter(SendMessagesHelper.SendingMediaInfo info) {
        final String footer = getFooter();
        if (TextUtils.isEmpty(footer)) {
            return;
        }
        final StringBuilder plain = new StringBuilder();
        final java.util.ArrayList<int[]> spans = new java.util.ArrayList<>();
        final java.util.ArrayList<String> urls = new java.util.ArrayList<>();
        final Matcher m = LINK.matcher(footer);
        int last = 0;
        while (m.find()) {
            plain.append(footer, last, m.start());
            final int off = plain.length();
            plain.append(m.group(1));
            spans.add(new int[]{off, m.group(1).length()});
            urls.add(m.group(2));
            last = m.end();
        }
        plain.append(footer.substring(last));
        final String base = info.caption == null ? "" : info.caption;
        if (base.endsWith(plain.toString())) {
            return;
        }
        final String sep = base.isEmpty() ? "" : "\n\n";
        final int shift = base.length() + sep.length();
        info.caption = base + sep + plain;
        if (info.entities == null) {
            info.entities = new java.util.ArrayList<>();
        }
        for (int i = 0; i < spans.size(); i++) {
            final TLRPC.TL_messageEntityTextUrl e = new TLRPC.TL_messageEntityTextUrl();
            e.offset = shift + spans.get(i)[0];
            e.length = spans.get(i)[1];
            e.url = urls.get(i);
            info.entities.add(e);
        }
    }

    /** Appends the footer to the caption. Idempotent. */
    public static String withFooter(String caption) {
        final String footer = getFooter();
        if (TextUtils.isEmpty(footer)) {
            return caption;
        }
        if (TextUtils.isEmpty(caption)) {
            return footer;
        }
        if (caption.endsWith(footer)) {
            return caption;
        }
        return caption + "\n\n" + footer;
    }

    // ---- output contract ----

    private static String buildSystemPrompt() {
        String prompt = prefs().getString(PREF_PROMPT, "");
        if (TextUtils.isEmpty(prompt)) {
            prompt = DEFAULT_PROMPT;
        }
        String template = prefs().getString(PREF_TEMPLATE, "");
        if (TextUtils.isEmpty(template)) {
            template = DEFAULT_TEMPLATE;
        }
        final StringBuilder sb = new StringBuilder(prompt.trim());
        sb.append("\n\nOUTPUT TEMPLATE (keep this exact structure and punctuation; replace every #placeholder with real tags describing the image, repeat or drop placeholders as needed): ").append(template.trim());
        final java.util.Set<String> known = getLinks().keySet();
        if (!known.isEmpty()) {
            sb.append("\n\nKNOWN TAGS (reuse these exact tags whenever they fit the image, invent new ones only when none fit): ").append(TextUtils.join(" ", known));
        }
        sb.append("\n\nOUTPUT CONTRACT (overrides any persona, style or instruction above): reply with that single line only. ")
            .append("Tags are '#' + lowercase letters/digits, '_' instead of spaces. The only other allowed characters are ( ) [ ] & / - . ")
            .append("At most ").append(getMaxTags()).append(" tags. No other words, no markdown, no emoji, no explanation, no line breaks.");
        return sb.toString();
    }

    private static String buildSystemPromptLegacy() {
        String prompt = prefs().getString(PREF_PROMPT, "");
        if (TextUtils.isEmpty(prompt)) {
            prompt = DEFAULT_PROMPT;
        }
        final int max = getMaxTags();
        return prompt.trim()
            + "\n\nOUTPUT CONTRACT (this overrides any persona, style or instruction above): "
            + "reply with hashtags only. Each tag starts with '#', is lowercase, uses '_' instead of spaces, "
            + "and tags are separated by a single space. At most " + max + " tags. "
            + "No other words, no punctuation, no emoji, no markdown, no quotes, no explanation, no line breaks.";
    }

    private static final Pattern TOKEN = Pattern.compile("#[\\p{L}\\p{N}_]+|[()\\[\\]&/\\-]");
    private static final Pattern LINK = Pattern.compile("\\[([^\\]]+)\\]\\(([^)\\s]+)\\)");

    /** tag -> url, one per line in settings: '#tag=https://...' */
    public static java.util.Map<String, String> getLinks() {
        final java.util.Map<String, String> out = new java.util.LinkedHashMap<>();
        for (String line : prefs().getString(PREF_LINKS, "").split("\\n")) {
            line = line.trim();
            if (line.isEmpty()) continue;
            int i = line.indexOf('=');
            if (i < 0) i = line.indexOf(' ');
            if (i <= 0) continue;
            String tag = line.substring(0, i).trim().toLowerCase();
            final String url = line.substring(i + 1).trim();
            if (!tag.startsWith("#")) tag = "#" + tag;
            if (!url.isEmpty()) out.put(tag, url);
        }
        return out;
    }

    /** Keeps only #tags and the template punctuation ( ) [ ] & / - ; everything else the model says is dropped. */
    public static String sanitize(String raw) {
        if (raw == null) {
            return "";
        }
        final int max = getMaxTags();
        final Set<String> seen = new LinkedHashSet<>();
        final StringBuilder sb = new StringBuilder();
        final Matcher m = TOKEN.matcher(raw);
        int prevEnd = -1;
        int tags = 0;
        while (m.find()) {
            String t = m.group();
            if (t.charAt(0) == '#') {
                t = t.toLowerCase();
                if (!seen.add(t)) {
                    prevEnd = m.end();
                    continue;
                }
                if (tags >= max) {
                    break;
                }
                tags++;
            }
            if (sb.length() > 0 && prevEnd >= 0 && m.start() > prevEnd) {
                sb.append(' ');
            }
            sb.append(t);
            prevEnd = m.end();
        }
        return tags == 0 ? "" : sb.toString().trim();
    }

    private static String sanitizeLegacy(String raw) {
        if (raw == null) {
            return "";
        }
        final int max = getMaxTags();
        final Set<String> tags = new LinkedHashSet<>();
        final Matcher m = TAG.matcher(raw);
        while (m.find() && tags.size() < max) {
            tags.add(m.group().toLowerCase());
        }
        if (tags.isEmpty()) {
            // model ignored the hash prefix: treat plain words as tags
            final String cleaned = raw.replaceAll("[^\\p{L}\\p{N}_\\s,]", " ");
            for (String w : cleaned.split("[\\s,]+")) {
                if (w.isEmpty()) continue;
                tags.add("#" + w.toLowerCase());
                if (tags.size() >= max) break;
            }
        }
        return TextUtils.join(" ", tags);
    }

    // ---- request ----

    private static String endpoint() {
        String base = prefs().getString(PREF_BASE_URL, "").trim();
        while (base.endsWith("/")) {
            base = base.substring(0, base.length() - 1);
        }
        return base.endsWith("/chat/completions") ? base : base + "/chat/completions";
    }

    private static String encodeImage(String path) throws Exception {
        final Bitmap bitmap = ImageLoader.loadBitmap(path, null, 1024, 1024, true);
        if (bitmap == null) {
            throw new Exception("image");
        }
        try {
            final ByteArrayOutputStream bos = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 85, bos);
            return "data:image/jpeg;base64," + Base64.encodeToString(bos.toByteArray(), Base64.NO_WRAP);
        } finally {
            bitmap.recycle();
        }
    }

    private static String shortError(Throwable e, int code, String body) {
        if (code == 401 || code == 403) return "AI: invalid API key";
        if (code == 404) return "AI: endpoint or model not found";
        if (code == 429) return "AI: rate limited";
        if (code >= 500) return "AI: server error " + code;
        if (code >= 400) return "AI: error " + code;
        if (e instanceof java.net.UnknownHostException || e instanceof java.net.SocketTimeoutException || e instanceof java.net.ConnectException) {
            return "AI: no connection";
        }
        if (e != null && "image".equals(e.getMessage())) return "AI: can't read image";
        return "AI: failed";
    }

    public static Job generate(String imagePath, Callback callback) {
        final Job job = new Job();
        if (!isConfigured()) {
            AndroidUtilities.runOnUIThread(() -> callback.onError("AI: set endpoint and key in Fork settings"));
            return job;
        }
        if (TextUtils.isEmpty(imagePath)) {
            AndroidUtilities.runOnUIThread(() -> callback.onError("AI: no image"));
            return job;
        }
        executor.submit(() -> {
            int code = 0;
            try {
                final String dataUrl = encodeImage(imagePath);
                final String model = prefs().getString(PREF_MODEL, DEFAULT_MODEL);

                final JSONObject text = new JSONObject().put("type", "text").put("text", "Tag this image.");
                final JSONObject image = new JSONObject().put("type", "image_url").put("image_url", new JSONObject().put("url", dataUrl));
                final JSONObject body = new JSONObject()
                    .put("model", TextUtils.isEmpty(model) ? DEFAULT_MODEL : model)
                    .put("stream", true)
                    .put("temperature", 0.3)
                    .put("max_tokens", 300)
                    .put("messages", new JSONArray()
                        .put(new JSONObject().put("role", "system").put("content", buildSystemPrompt()))
                        .put(new JSONObject().put("role", "user").put("content", new JSONArray().put(text).put(image))));

                final HttpURLConnection conn = (HttpURLConnection) new URL(endpoint()).openConnection();
                job.conn = conn;
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Authorization", "Bearer " + prefs().getString(PREF_KEY, "").trim());
                conn.setRequestProperty("Content-Type", "application/json");
                conn.setRequestProperty("Accept", "text/event-stream");
                conn.setConnectTimeout(20000);
                conn.setReadTimeout(60000);
                conn.setDoOutput(true);
                try (OutputStream os = conn.getOutputStream()) {
                    os.write(body.toString().getBytes("utf-8"));
                }

                code = conn.getResponseCode();
                if (code < 200 || code >= 300) {
                    final String msg = shortError(null, code, null);
                    AndroidUtilities.runOnUIThread(() -> callback.onError(msg));
                    return;
                }

                final StringBuilder acc = new StringBuilder();
                final StringBuilder raw = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "utf-8"))) {
                    String line;
                    while (!job.cancelled && (line = br.readLine()) != null) {
                        if (!line.startsWith("data:")) {
                            raw.append(line);
                            continue;
                        }
                        final String data = line.substring(5).trim();
                        if (data.isEmpty()) continue;
                        if ("[DONE]".equals(data)) break;
                        try {
                            final JSONArray choices = new JSONObject(data).optJSONArray("choices");
                            if (choices == null || choices.length() == 0) continue;
                            final JSONObject delta = choices.getJSONObject(0).optJSONObject("delta");
                            if (delta == null) continue;
                            final String piece = delta.optString("content", "");
                            if (piece.isEmpty() || "null".equals(piece)) continue;
                            acc.append(piece);
                            final String shown = sanitize(acc.toString());
                            AndroidUtilities.runOnUIThread(() -> {
                                if (!job.cancelled) callback.onDelta(shown);
                            });
                        } catch (Exception ignore) {}
                    }
                }
                if (job.cancelled) {
                    return;
                }
                if (acc.length() == 0 && raw.length() > 0) {
                    // server ignored stream=true and returned plain json
                    try {
                        final JSONObject o = new JSONObject(raw.toString());
                        final JSONArray choices = o.optJSONArray("choices");
                        if (choices != null && choices.length() > 0) {
                            acc.append(choices.getJSONObject(0).getJSONObject("message").optString("content", ""));
                        }
                    } catch (Exception ignore) {}
                }
                final String result = sanitize(acc.toString());
                AndroidUtilities.runOnUIThread(() -> {
                    if (TextUtils.isEmpty(result)) {
                        callback.onError("AI: empty result");
                    } else {
                        callback.onDone(result);
                    }
                });
            } catch (Throwable e) {
                if (job.cancelled) {
                    return;
                }
                FileLog.e(e);
                final String msg = shortError(e, code, null);
                AndroidUtilities.runOnUIThread(() -> callback.onError(msg));
            }
        });
        return job;
    }
}
