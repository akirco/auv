#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D texY;
uniform sampler2D texU;
uniform sampler2D texV;
uniform vec4 cmat; // BT.601/709 系数：(rv, gu, gv, bu)，由 CPU 按分辨率设置
void main() {
    float y = texture(texY, TexCoord).r;
    float u = texture(texU, TexCoord).r - 0.5;
    float v = texture(texV, TexCoord).r - 0.5;
    float r = y + cmat.x * v;
    float g = y + cmat.y * u + cmat.z * v;
    float b = y + cmat.w * u;
    FragColor = vec4(r, g, b, 1.0);
}
