#version 330 core
// 时间 OSD：顶点已是 NDC 坐标（由 CPU 按窗口物理像素换算），原样输出
layout (location = 0) in vec2 aPos;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}
