# taffy Mesh 编辑器 1.1

用于打开 PLY 的本地桌面工具，保留 taffy 编辑器的视图、图层、XYZ 范围、相机和截图功能。Mesh 编辑器不提供点云滤波、删点和补点，也不执行 PCD 到网格的重建。

## 安装与启动

[下载 taffy-mesh-editor_1.1_all.deb](https://github.com/LYJX-robotic-CC/taffy-editor/releases/download/v1.3/taffy-mesh-editor_1.1_all.deb)，适用于 Ubuntu 22.04 / Python 3.10。

```bash
sudo apt install ./taffy-mesh-editor_1.1_all.deb
taffy-mesh-editor
```

也可从应用菜单打开“taffy Mesh编辑器”。依赖 PyQt5、NumPy、VTK 和支持 OpenGL 3.2 或更高版本的显卡驱动。安装包包含应用字节码，不含 Python 明文源码。

## 使用

1. 点击左侧“导入 PLY 文件”，支持 ASCII 和 binary PLY。下次导入会记住上次选择的位置。
2. 多个文件作为独立图层加入场景。单击选中；双击会显示并选中。左侧眼睛控制显隐。
3. 右侧 `⋯` 菜单支持导出、另存 PLY 副本及删除图层。删除仅移除场景中的图层，不删除原文件。长文件名自动省略，悬停可查看完整名称。
4. 鼠标左键拖动旋转、中键平移、右键缩放。可一键俯视、按住按钮绕 Z 轴旋转，以及显示网格。
5. 左侧 Z、右侧 X 和底部 Y 滑块裁剪当前层显示范围。“重置”恢复 XYZ 全部范围。
6. 可切换表面、线框或点显示，调整点大小，并导出截图。

导出 PLY 使用完整网格，显示范围裁剪不会截掉导出数据。顶部和底部显示顶点数、面数和坐标范围。大模型直接交给 VTK 渲染，加载和导出可能耗时。

## 1.1 更新

- 空白场景使用浅色背景和网格。
- 新增从场景移除图层，保留磁盘原文件。
- 长文件名省略显示，始终保留右侧菜单空间。
- 记住最近选择的文件和目录，重启软件后仍可恢复。
