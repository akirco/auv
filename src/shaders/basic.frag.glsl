#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D texY;
uniform sampler2D texU;
uniform sampler2D texV;
uniform vec4 cmat;         // BT.601/709/2020 系数：(rv, gu, gv, bu)
uniform vec2 yParams;      // 预合并：Y' = raw * scale + bias
uniform vec2 uvParams;     // 预合并：U'/V' = raw * scale + bias
uniform int hdrMode;       // 0=SDR, 1=PQ(HDR10), 2=HLG
uniform int hdrPrimaries;  // 1 = BT.2020 -> BT.709
uniform int semiPlanar;    // 1 = NV12/P010（UV 交织在 texU.rg）

// SMPTE ST 2084 (PQ) EOTF，输出线性光 0..1
vec3 pq_eotf(vec3 e) {
    const float m1 = 0.15930176;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.851562;
    const float c3 = 18.6875;
    vec3 e1 = pow(clamp(e, 0.0, 1.0), vec3(1.0 / m2));
    vec3 num = max(e1 - c1, 0.0);
    vec3 den = c2 - c3 * e1;
    return pow(num / max(den, vec3(1e-6)), vec3(1.0 / m1));
}

// ARIB STD-B67 (HLG) 逆 OETF
vec3 hlg_inv(vec3 e) {
    const float a = 0.17883277;
    const float b = 0.28466892;
    const float c = 0.5599107;
    vec3 lo = e * e / 3.0;
    vec3 hi = (exp((e - c) / a) + b) / 12.0;
    return mix(lo, hi, step(vec3(0.5), e));
}

// Hable filmic tone curve（只按亮度算一次，RGB 等比缩放）
float hable(float x) {
    const float A = 0.15, B = 0.50, C = 0.10;
    const float D = 0.20, E = 0.02, F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

void main() {
    float y = texture(texY, TexCoord).r * yParams.x + yParams.y;
    float u;
    float v;
    if (semiPlanar == 1) {
        vec2 uv = texture(texU, TexCoord).rg * uvParams.x + uvParams.y;
        u = uv.r;
        v = uv.g;
    } else {
        u = texture(texU, TexCoord).r * uvParams.x + uvParams.y;
        v = texture(texV, TexCoord).r * uvParams.x + uvParams.y;
    }
    vec3 rgb = vec3(y + cmat.x * v, y + cmat.y * u + cmat.z * v, y + cmat.w * u);

    if (hdrMode == 0) {
        FragColor = vec4(rgb, 1.0);
        return;
    }

    rgb = clamp(rgb, 0.0, 1.0);
    vec3 lin = (hdrMode == 1) ? pq_eotf(rgb) : pow(hlg_inv(rgb), vec3(1.2));
    if (hdrPrimaries == 1) {
        // BT.2020 -> BT.709（线性光）
        lin = vec3(
             1.6605 * lin.r - 0.5876 * lin.g - 0.0728 * lin.b,
            -0.1246 * lin.r + 1.1329 * lin.g - 0.0083 * lin.b,
            -0.0182 * lin.r - 0.1006 * lin.g + 1.1187 * lin.b
        );
    }
    lin = clamp(lin, 0.0, 1.0);

    // 亮度 tone map：一次 hable + 等比缩放；比逐通道 Hable 快且保持色相
    float lum = max(dot(lin, vec3(0.2126, 0.7152, 0.0722)), 1e-6);
    float mapped_lum = hable(lum) / hable(11.2);
    vec3 mapped = clamp(lin * (mapped_lum / lum), 0.0, 1.0);
    FragColor = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}
