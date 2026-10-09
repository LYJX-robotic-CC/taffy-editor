# taffy 编辑器

taffy 编辑器是一款面向 PCD 点云的本地桌面工具，支持点云导入、三维浏览、XYZ 范围裁剪、显示滤波、区域删除和区域补全。应用使用 Qt 和 VTK 构建，不依赖浏览器。

## 功能

- 导入和显示 PCD 点云，支持大规模点云抽样渲染
- 鼠标常规三维交互：左键旋转、中键平移、右键缩放
- X、Y、Z 三轴可视范围滑块
- 一键俯视、绕 Z 轴旋转和网格显示
- 体素、统计离群点和半径离群点滤波
- 滤波只处理当前可视区域，点击“重新计算滤波”后才应用参数
- 双击选择当前可视区域，计算三维凸包并删除或随机近似补全
- 图层可见性、图层选择、PCD 导出和原始 PCD 保存

## 安装

系统要求：Debian/Ubuntu Linux 和图形桌面环境。

```bash
sudo dpkg -i taffy-editor_1.0_all.deb
sudo apt-get -f install
```

安装后可从应用菜单启动，或运行 `taffy-editor`。软件包依赖 PyQt5、VTK、Open3D、NumPy 和 SciPy。

## 从源码运行

```bash
python3 main.py
./启动taffy编辑器.sh
```

启动后默认是空白场景，需要通过“导入 PCD”载入点云。

## 区域编辑流程

1. 导入并选中图层，调整 X、Y、Z 滑块到目标范围。
2. 点击“开始选点区域”，在当前可视化区域内双击至少 4 个不共面的点。
3. 点击“计算凸包区域”，设置厘米级 XYZ 外扩距离后重新计算。
4. 点击“删除区域点”或“补全区域点”。补全采用随机近似采样，保持大致密度并避免占用过多内存。

## 开发与打包

```bash
python3 -m py_compile main.py
dpkg-deb --build --root-owner-group build-deb/taffy-editor-1.0 dist/taffy-editor_1.0_all.deb
```

## 项目结构

```text
main.py                         主程序
assets/tafei-pcd-editor.png     应用图标
启动taffy编辑器.sh              源码启动脚本
```

当前版本暂未指定开源许可证。
