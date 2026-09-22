#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D texY;
uniform sampler2D texU;
uniform sampler2D texV;
uniform vec4 cmat;   // BT.601/709/2020 系数：(rv, gu, gv, bu)，由 CPU 按流的色彩信息设置
uniform vec2 rangeY;   // Y 的 range 归一：(offset, gain)，limited=[16/255, 255/219]，full=[0, 1]
uniform vec2 rangeUV;  // U/V 的 range 归一：(center, gain)，limited=[0.5, 255/224]，full=[0.5, 1]
void main() {
    float y = (texture(texY, TexCoord).r - rangeY.x) * rangeY.y;
    float u = (texture(texU, TexCoord).r - rangeUV.x) * rangeUV.y;
    float v = (texture(texV, TexCoord).r - rangeUV.x) * rangeUV.y;
    float r = y + cmat.x * v;
    float g = y + cmat.y * u + cmat.z * v;
    float b = y + cmat.w * u;
    FragColor = vec4(r, g, b, 1.0);
}