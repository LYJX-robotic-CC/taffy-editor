#!/usr/bin/env python3
import sys
import shutil
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import vtk
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError, cKDTree
from PyQt5.QtCore import QElapsedTimer, QEvent, QPoint, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygon
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QStyle, QStyleOptionSlider, QToolButton, QVBoxLayout, QWidget, QMenu,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk


APP_STYLE = """
QMainWindow, QWidget { background: #f3f5f4; color: #1f2b2b; font-family: 'Inter', 'Noto Sans CJK SC', sans-serif; font-size: 13px; }
QLabel { background: transparent; border: none; }
QFrame#appShell { background: #fff8fd; border: 1px solid #eaddeb; border-radius: 18px; }
QFrame#header { background: #fff5fb; border-bottom: 1px solid #eaddeb; border-top-left-radius: 18px; border-top-right-radius: 18px; }
QFrame#sidebar { background: #fbf8ff; border-right: 1px solid #e7dfef; }
QFrame#toolbar { background: #fffafd; border-bottom: 1px solid #eee4ed; }
QFrame#filterpanel { background: #fffafd; border-top: 1px solid #eee4ed; }
QFrame#rightpanel { background: #fbfaff; border-left: 1px solid #e7dfef; }
QToolButton#processHeader { border: 1px solid #eadde8; background: #fffafd; color: #735d72; border-radius: 8px; padding: 8px 10px; text-align: left; font-weight: 650; }
QToolButton#processHeader:hover { background: #fff0f7; border-color: #e4a9c8; color: #a53a70; }
QLabel#brand { color: #a73d70; font-size: 16px; font-weight: 700; }
QLabel#brandAccent { color: #9b80a1; font-weight: 600; }
QLabel#eyebrow { color: #a38ca1; font-size: 10px; font-weight: 700; letter-spacing: 1px; }
QLabel#muted { color: #928492; font-size: 11px; }
QLabel#metricValue { color: #663e69; font-size: 17px; font-weight: 700; }
QLabel#metricCaption { color: #aa98a8; font-size: 10px; }
QLabel#sectionTitle { color: #6a4d6d; font-weight: 650; font-size: 13px; }
QLabel#filename { color: #6e5b6f; font-size: 12px; }
QPushButton { border: 1px solid #eadde8; background: #fffefe; color: #735d72; border-radius: 9px; padding: 7px 12px; }
QPushButton:hover { background: #fff0f7; border-color: #e4a9c8; color: #a53a70; }
QPushButton#primary { background: #f8dce9; border-color: #e4aac4; color: #7f3159; font-weight: 700; padding: 9px 13px; }
QPushButton#primary:hover { background: #f2c9dc; border-color: #d98bad; color: #692445; }
QPushButton#flat { border: none; background: transparent; padding: 6px; color: #8e758d; border-radius: 8px; }
QPushButton#flat:hover { background: #f8e8f3; color: #a53a70; }
QPushButton#nav { text-align: left; border: none; background: transparent; padding: 10px; color: #806f83; border-radius: 8px; }
QPushButton#nav[active="true"] { color: #a23e70; background: #fce9f3; font-weight: 650; }
QPushButton#nav:hover { background: #f8edf7; }
QToolButton#windowControl { border: none; border-radius: 9px; color: #886f85; background: transparent; font-size: 15px; }
QToolButton#windowControl:hover { background: #f7ddeb; color: #a53367; }
QToolButton#windowClose:hover { background: #e86f8e; color: white; }
QLineEdit, QDoubleSpinBox { background: transparent; border: none; padding: 2px 0; color: #5b4a5d; }
QSlider::groove:horizontal { height: 3px; background: #eadfea; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 7px; background: #d96196; border: 2px solid #ffffff; }
QCheckBox { color: #655368; spacing: 7px; padding: 4px 0; background: transparent; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #d9c8d8; border-radius: 5px; background: transparent; }
QCheckBox::indicator:checked { background: #d96196; border-color: #d96196; }
QMenu { background: #fffaff; border: 1px solid #e6d9e6; border-radius: 8px; padding: 5px; color: #614e63; }
QMenu::item { padding: 7px 24px 7px 10px; border-radius: 5px; }
QMenu::item:selected { background: #f9e7f1; color: #a43e70; }
"""

APP_VERSION = '1.0'


def fmt_number(value):
    return f"{value:,.0f}"


def make_demo_cloud():
    rng = np.random.default_rng(18)
    count = 52000
    x = rng.uniform(-42, 42, count)
    y = rng.uniform(-31, 31, count)
    road = (np.abs(y - 2) < 2.4) | (np.abs(x + 8) < 2.1)
    terrain_z = 0.32 * np.sin(x / 8) + 0.22 * np.cos(y / 7) + rng.normal(0, 0.09, count)
    ground = np.column_stack((x[~road], y[~road], terrain_z[~road]))
    parts = [ground]
    buildings = [(-29, -20, -22, -12, 8), (-16, -7, -22, -14, 13), (1, 10, -23, -14, 10),
                 (18, 28, -21, -11, 16), (-29, -20, 12, 21, 12), (-12, -3, 13, 22, 18),
                 (5, 15, 13, 22, 9), (23, 32, 10, 20, 14)]
    for x0, x1, y0, y1, height in buildings:
        n = 1400
        face = rng.integers(0, 4, n)
        bx = rng.uniform(x0, x1, n)
        by = rng.uniform(y0, y1, n)
        bz = rng.uniform(0.15, height, n)
        bx = np.where(face == 0, x0, np.where(face == 1, x1, bx))
        by = np.where(face == 2, y0, np.where(face == 3, y1, by))
        parts.append(np.column_stack((bx, by, bz)))
        roof_count = 350
        parts.append(np.column_stack((rng.uniform(x0, x1, roof_count), rng.uniform(y0, y1, roof_count), rng.normal(height, .08, roof_count))))
    return np.concatenate(parts).astype(np.float32)


MAX_RENDER_POINTS = 1_500_000


def load_pcd_data(path):
    header = {}
    with open(path, 'rb') as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('PCD 文件头不完整。')
            parts = line.decode('ascii', errors='strict').strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == 'DATA':
                data_offset = stream.tell()
                break

    fields = header.get('FIELDS', [])
    point_count = int(header.get('POINTS', [0])[0])
    if not point_count:
        point_count = int(header.get('WIDTH', ['0'])[0]) * int(header.get('HEIGHT', ['1'])[0])
    if header.get('DATA') == ['binary'] and all(key in fields for key in ('x', 'y', 'z')):
        sizes = [int(v) for v in header.get('SIZE', [])]
        types = header.get('TYPE', [])
        counts = [int(v) for v in header.get('COUNT', ['1'] * len(fields))]
        type_map = {('F', 4): '<f4', ('F', 8): '<f8', ('I', 1): 'i1', ('I', 2): '<i2', ('I', 4): '<i4', ('I', 8): '<i8', ('U', 1): 'u1', ('U', 2): '<u2', ('U', 4): '<u4', ('U', 8): '<u8'}
        dtype_fields = []
        for name, size, kind, count in zip(fields, sizes, types, counts):
            dtype = type_map.get((kind, size))
            if dtype is None:
                break
            dtype_fields.append((name, dtype) if count == 1 else (name, dtype, (count,)))
        else:
            records = np.memmap(path, mode='r', dtype=np.dtype(dtype_fields), offset=data_offset, shape=(point_count,))
            stride = max(1, int(np.ceil(point_count / MAX_RENDER_POINTS)))
            display_points = np.column_stack((records['x'][::stride], records['y'][::stride], records['z'][::stride])).astype(np.float32, copy=False)
            return display_points, records['z'], point_count

    cloud = o3d.io.read_point_cloud(str(path), remove_nan_points=True, remove_infinite_points=True)
    all_points = np.asarray(cloud.points)
    if not len(all_points):
        raise ValueError('文件中没有有效 XYZ 点。')
    total_points = len(all_points)
    full_z = all_points[:, 2].copy()
    stride = max(1, int(np.ceil(total_points / MAX_RENDER_POINTS)))
    display_points = np.ascontiguousarray(all_points[::stride, :3], dtype=np.float32)
    return display_points, full_z, total_points


def load_full_xyz(path):
    """Load full XYZ from a PCD for editing, using memmap for binary files."""
    header = {}
    with open(path, 'rb') as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('PCD 文件头不完整。')
            parts = line.decode('ascii', errors='strict').strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == 'DATA':
                offset = stream.tell()
                break
    fields = header.get('FIELDS', [])
    if not all(name in fields for name in ('x', 'y', 'z')):
        raise ValueError('PCD 中缺少 XYZ 字段。')
    count = int(header.get('POINTS', [0])[0])
    if header.get('DATA') == ['binary']:
        sizes = [int(value) for value in header.get('SIZE', [])]
        types = header.get('TYPE', [])
        counts = [int(value) for value in header.get('COUNT', ['1'] * len(fields))]
        type_map = {('F', 4): '<f4', ('F', 8): '<f8', ('I', 1): 'i1', ('I', 2): '<i2', ('I', 4): '<i4', ('I', 8): '<i8', ('U', 1): 'u1', ('U', 2): '<u2', ('U', 4): '<u4', ('U', 8): '<u8'}
        dtype_fields = []
        for name, size, kind, item_count in zip(fields, sizes, types, counts):
            dtype = type_map.get((kind, size))
            if dtype is None:
                raise ValueError(f'不支持 PCD 字段类型：{name}')
            dtype_fields.append((name, dtype) if item_count == 1 else (name, dtype, (item_count,)))
        records = np.memmap(path, mode='r', dtype=np.dtype(dtype_fields), offset=offset, shape=(count,))
        return np.column_stack((records['x'], records['y'], records['z'])).astype(np.float32)
    cloud = o3d.io.read_point_cloud(str(path), remove_nan_points=True, remove_infinite_points=True)
    return np.asarray(cloud.points, dtype=np.float32)


class HeightRangeSlider(QWidget):
    rangeChanged = pyqtSignal(int, int)

    def __init__(self, orientation=Qt.Vertical, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.setStyleSheet('background:#fff8fd;border:1px solid #eaddea;border-radius:8px;')
        self.setFocusPolicy(Qt.StrongFocus)
        self.low = 0
        self.high = 1000
        self.dragging = None
        if orientation == Qt.Vertical:
            self.setFixedWidth(32)
            self.setMinimumHeight(140)
        else:
            self.setFixedHeight(32)
            self.setMinimumWidth(180)

    def _track(self):
        extent = self.height() if self.orientation == Qt.Vertical else self.width()
        start, end = 12, extent - 12
        return start, max(start + 1, end)

    def _y(self, value):
        start, end = self._track()
        if self.orientation == Qt.Vertical:
            return end - (end - start) * value / 1000
        return start + (end - start) * value / 1000

    def _value(self, coordinate):
        start, end = self._track()
        if self.orientation == Qt.Vertical:
            value = (end - coordinate) / (end - start) * 1000
        else:
            value = (coordinate - start) / (end - start) * 1000
        return int(np.clip(round(value), 0, 1000))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        start, end = self._track()
        if self.orientation == Qt.Vertical:
            center = self.width() // 2
            point = lambda value: (center, round(self._y(value)))
        else:
            center = self.height() // 2
            point = lambda value: (round(self._y(value)), center)
        a, b = point(0), point(1000)
        painter.setPen(QPen(QColor('#c9b8cc'), 6, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(*a, *b)
        painter.setPen(QPen(QColor('#d96196'), 3, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(*point(self.low), *point(self.high))
        for value in (self.high, self.low):
            painter.setPen(QPen(QColor('#d9a8bf'), 1))
            painter.setBrush(QColor('#ffffff'))
            x, y = point(value)
            painter.drawEllipse(x - 6, y - 6, 12, 12)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor('#9d3b6c'))
            painter.drawEllipse(x - 3, y - 3, 6, 6)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        value = self._value(event.y() if self.orientation == Qt.Vertical else event.x())
        self.dragging = 'low' if abs(value - self.low) < abs(value - self.high) else 'high'
        self._set_drag_value(value)

    def mouseMoveEvent(self, event):
        if self.dragging and event.buttons() & Qt.LeftButton:
            self._set_drag_value(self._value(event.y() if self.orientation == Qt.Vertical else event.x()))

    def mouseReleaseEvent(self, event):
        self.dragging = None

    def keyPressEvent(self, event):
        if event.key() not in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            return super().keyPressEvent(event)
        self.dragging = 'low' if self.dragging != 'high' else 'high'
        delta = 10 if event.key() in (Qt.Key_Up, Qt.Key_Right) else -10
        self._set_drag_value((self.low if self.dragging == 'low' else self.high) + delta)

    def _set_drag_value(self, value):
        if self.dragging == 'low':
            self.low = min(max(value, 0), self.high - 1)
        elif self.dragging == 'high':
            self.high = max(min(value, 1000), self.low + 1)
        self.update()
        self.rangeChanged.emit(self.low, self.high)

    def set_range(self, low, high):
        self.low, self.high = low, high
        self.update()


class DragOnlySlider(QSlider):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._dragging_handle = False

    def wheelEvent(self, event):
        event.ignore()

    def mousePressEvent(self, event):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        handle = self.style().subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderHandle, self)
        if event.button() != Qt.LeftButton or not handle.adjusted(-5, -5, 5, 5).contains(event.pos()):
            event.ignore()
            return
        self._dragging_handle = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_handle:
            super().mouseMoveEvent(event)
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if self._dragging_handle:
            super().mouseReleaseEvent(event)
        self._dragging_handle = False


class LayerSelectButton(QPushButton):
    doubleClicked = pyqtSignal()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class EmbeddedVTKWidget(QVTKRenderWindowInteractor):
    doubleClicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
            self.doubleClicked.emit(event.x(), self.height() - event.y() - 1)
        super().mousePressEvent(event)

    def winId(self):
        self.setAttribute(Qt.WA_DontCreateNativeAncestors, True)
        return super().winId()


class TaffyPatternFrame(QFrame):
    """A very light cat-ear and star pattern used behind tool controls."""
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(217, 97, 150, 28), 1.2))
        painter.setBrush(Qt.NoBrush)
        for row, y in enumerate(range(22, self.height() + 55, 72)):
            offset = 16 if row % 2 else 58
            for x in range(offset, self.width() + 55, 94):
                painter.drawEllipse(x, y + 7, 12, 10)
                painter.drawPolyline(QPolygon([QPoint(x + 1, y + 9), QPoint(x + 4, y), QPoint(x + 7, y + 8)]))
                painter.drawPolyline(QPolygon([QPoint(x + 6, y + 8), QPoint(x + 10, y), QPoint(x + 12, y + 9)]))
                painter.drawLine(x + 22, y + 7, x + 28, y + 7)
                painter.drawLine(x + 25, y + 4, x + 25, y + 10)
                painter.drawEllipse(x + 43, y + 4, 3, 3)


class DisplayFilterWorker(QThread):
    completed = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, request_id, points, voxel_size, statistical, stat_neighbors, std_ratio,
                 radius, radius_neighbors, radius_size):
        super().__init__()
        self.request_id = request_id
        self.points = points
        self.voxel_size = voxel_size
        self.statistical = statistical
        self.stat_neighbors = stat_neighbors
        self.std_ratio = std_ratio
        self.radius = radius
        self.radius_neighbors = radius_neighbors
        self.radius_size = radius_size

    def run(self):
        try:
            points = self.points
            if self.voxel_size > 0:
                # One representative per voxel keeps the displayed density uniform.
                cells = np.floor(points / self.voxel_size).astype(np.int32)
                _, indices = np.unique(cells, axis=0, return_index=True)
                points = np.ascontiguousarray(points[np.sort(indices)])
            if self.isInterruptionRequested():
                return
            if self.statistical or self.radius:
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points)
                if self.statistical:
                    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=self.stat_neighbors, std_ratio=self.std_ratio)
                if self.radius and len(cloud.points):
                    cloud, _ = cloud.remove_radius_outlier(nb_points=self.radius_neighbors, radius=self.radius_size)
                points = np.asarray(cloud.points, dtype=np.float32)
            if not self.isInterruptionRequested():
                self.completed.emit(self.request_id, points)
        except Exception as error:
                self.failed.emit(self.request_id, str(error))


class FullPointLoadWorker(QThread):
    loaded = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, request_id, path):
        super().__init__()
        self.request_id = request_id
        self.path = path

    def run(self):
        try:
            points = load_full_xyz(self.path)
            points = np.ascontiguousarray(points[np.isfinite(points).all(axis=1)], dtype=np.float32)
            self.loaded.emit(self.request_id, points)
        except Exception as error:
            self.failed.emit(self.request_id, str(error))


class PointCloudWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('taffy编辑器')
        self.setWindowIcon(QIcon(str(Path(__file__).parent / 'assets' / 'tafei-pcd-editor.png')))
        self.resize(1440, 920)
        self.setMinimumSize(1180, 700)
        self.points = None
        self.render_source_points = None
        self.full_z = None
        self.total_points = 0
        self.layers = []
        self.layer_rows = {}
        self.active_layer_id = None
        self.next_layer_id = 1
        self._window_drag_offset = None
        self.z_min, self.z_max = -1.2, 8.4
        self.grid_visible = True
        self.orthographic = False
        self.point_size = 3.0
        self.rotate_direction = 0
        self.rotate_clock = QElapsedTimer()
        self.rotate_timer = QTimer(self)
        self.rotate_timer.setInterval(16)
        self.rotate_timer.timeout.connect(self._rotate_held_view)
        self.filter_request_id = 0
        self.filter_recalculate_requested = False
        self.filter_worker = None
        self.filter_scope = None
        self.filter_result_popup = None
        self.point_edit_mode = None
        self.region_pick_points = []
        self.region_pick_scope = None
        self.region_marker_actors = []
        self.selected_hull_equations = None
        self.selected_region_vertices = None
        self.region_preview_actor = None
        self.edit_load_operation = None
        self.completion_base_points = None
        self.completion_markers = []
        self.last_completion_click = None
        self.last_edit_click_time = 0.0
        self.last_edit_click_pos = None
        self.edit_load_worker = None
        self.edit_load_request = 0
        self.filter_debounce = QTimer(self)
        self.filter_debounce.setSingleShot(True)
        self.filter_debounce.setInterval(260)
        self.filter_debounce.timeout.connect(self.start_display_filter)
        self._build_ui()
        self._build_vtk()
        # Start with an empty scene. Point clouds are loaded explicitly via
        # the import action so launching the editor never reads a large file.
        self.header_file.setText('未导入点云')
        self.layer_name.setText('等待导入')
        self.layer_points.setText('0 点')
        self.metric_points.setText('0')
        self.metric_height.setText('--')
        self.visible_count.setText('0')
        self.visible_percent.setText('0% 点云')
    def eventFilter(self, watched, event):
        if watched.objectName() == 'header':
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._window_drag_offset = event.globalPos() - self.frameGeometry().topLeft()
                return True
            if event.type() == QEvent.MouseMove and self._window_drag_offset is not None and event.buttons() & Qt.LeftButton:
                self.move(event.globalPos() - self._window_drag_offset)
                return True
            if event.type() == QEvent.MouseButtonRelease:
                self._window_drag_offset = None
                return True
        return super().eventFilter(watched, event)

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        header = TaffyPatternFrame(objectName='header')
        header.installEventFilter(self)
        header.setFixedHeight(58)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(18, 0, 13, 0)
        hl.setSpacing(11)
        mark = QLabel()
        mark.setPixmap(QPixmap(str(Path(__file__).parent / 'assets' / 'tafei-pcd-editor.png')).scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        mark.setFixedSize(32, 32)
        brand = QLabel('taffy编辑器')
        brand.setObjectName('brand')
        accent = QLabel('本地版')
        accent.setObjectName('brandAccent')
        hl.addWidget(mark)
        hl.addWidget(brand)
        hl.addWidget(accent)
        hl.addSpacing(16)
        self.header_file = QLabel('城市地形样本')
        self.header_file.setObjectName('filename')
        hl.addWidget(QLabel('●'))
        hl.itemAt(hl.count() - 1).widget().setStyleSheet('color:#d96196;font-size:8px')
        hl.addWidget(self.header_file)
        hl.addStretch()
        self.metric_points, self.metric_height = QLabel('84,216'), QLabel('−1.2 — 8.4 m')
        for caption, value in (('点云数量', self.metric_points), ('高程范围', self.metric_height)):
            block = QVBoxLayout()
            block.setSpacing(0)
            value.setObjectName('metricValue')
            label = QLabel(caption)
            label.setObjectName('metricCaption')
            block.addWidget(value, alignment=Qt.AlignRight)
            block.addWidget(label, alignment=Qt.AlignRight)
            hl.addLayout(block)
            hl.addSpacing(24)
        self._button('导出截图', 'primary', self.export_image, hl)
        root_layout.addWidget(header)

        body = QHBoxLayout()
        body.setSpacing(0)
        sidebar = TaffyPatternFrame(objectName='sidebar')
        sidebar.setFixedWidth(230)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(14, 22, 14, 15)
        sl.setSpacing(8)
        section = QLabel('工作区')
        section.setObjectName('eyebrow')
        sl.addWidget(section)
        nav = QPushButton('◈   点云查看器')
        nav.setObjectName('nav')
        nav.setProperty('active', True)
        sl.addWidget(nav)
        self._button('＋   导入 PCD 文件', 'primary', self.open_file, sl)
        sl.addSpacing(25)
        row = QHBoxLayout()
        layers = QLabel('场景图层')
        layers.setObjectName('eyebrow')
        row.addWidget(layers)
        row.addStretch()
        row.addWidget(QLabel('+'))
        sl.addLayout(row)
        sl.addSpacing(6)
        self.layers_container = QWidget()
        self.layers_container.setStyleSheet('background:transparent')
        self.layers_layout = QVBoxLayout(self.layers_container)
        self.layers_layout.setContentsMargins(0, 0, 0, 0)
        self.layers_layout.setSpacing(3)
        sl.addWidget(self.layers_container)
        self.layer_name = QLabel('urban_scan.pcd')
        self.layer_points = QLabel('84,216 点   ·   XYZ')
        self.layer_points.setObjectName('muted')
        sl.addWidget(self.layer_points)
        sl.addStretch()
        local = QLabel('▣   本地工作区')
        local.setStyleSheet('font-weight:600;color:#7d5176;padding-top:10px')
        sl.addWidget(local)
        privacy = QLabel('文件仅在此设备处理')
        privacy.setObjectName('muted')
        sl.addWidget(privacy)
        body.addWidget(sidebar)

        main = QWidget()
        ml = QVBoxLayout(main)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        toolbar = QFrame(objectName='toolbar')
        toolbar.setFixedHeight(43)
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(20, 0, 17, 0)
        crumb = QLabel('项目   ›   点云查看器')
        crumb.setStyleSheet('color:#99859c;font-size:11px')
        tl.addWidget(crumb)
        tl.addStretch()
        self._button('适配视图', 'flat', self.reset_camera, tl)
        self._button('网格 ✓', 'flat', self.toggle_grid, tl)
        ml.addWidget(toolbar)

        view = QFrame()
        self.view_container = view
        view.setStyleSheet('background:#fff8fd;')
        vl = QVBoxLayout(view)
        vl.setContentsMargins(4, 4, 4, 4)
        vl.setSpacing(3)
        viewhead = QHBoxLayout()
        name = QLabel('●  三维视图')
        name.setStyleSheet('font-size:12px;font-weight:650;color:#714c70')
        self.view_label = QLabel('俯视  ·  Z 轴高度着色')
        self.view_label.setObjectName('muted')
        viewhead.addWidget(name)
        viewhead.addSpacing(8)
        viewhead.addWidget(self.view_label)
        viewhead.addStretch()
        self._button('俯视', 'flat', self.set_top_view, viewhead).setToolTip('一键俯视')
        self._hold_rotate_button('↺', '按住向左绕 Z 轴旋转（10°/秒）', -1, viewhead)
        self._hold_rotate_button('↻', '按住向右绕 Z 轴旋转（10°/秒）', 1, viewhead)
        self._button('透视', 'flat', self.toggle_projection, viewhead).setToolTip('切换透视 / 正交投影')
        self._button('⛶', 'flat', self.toggle_fullscreen, viewhead)
        vl.addLayout(viewhead)

        self.vtk_widget = EmbeddedVTKWidget(view)
        self.vtk_widget.doubleClicked.connect(self._on_embedded_double_click)
        self.vtk_widget.setMinimumHeight(360)
        self.vtk_widget.setMinimumWidth(420)

        self.z_range_slider = HeightRangeSlider(Qt.Vertical)
        self.z_range_slider.setMaximumHeight(152)
        self.z_range_slider.rangeChanged.connect(lambda low, high: self.apply_axis_range('z', low, high))
        self.x_range_slider = HeightRangeSlider(Qt.Vertical)
        self.x_range_slider.setMaximumHeight(152)
        self.x_range_slider.rangeChanged.connect(lambda low, high: self.apply_axis_range('x', low, high))
        self.y_range_slider = HeightRangeSlider(Qt.Horizontal)
        self.y_range_slider.setFixedWidth(240)
        self.y_range_slider.rangeChanged.connect(lambda low, high: self.apply_axis_range('y', low, high))
        self.range_slider = self.z_range_slider

        self.z_overlay_label = QLabel('Z  高度')
        self.z_overlay_label.setObjectName('eyebrow')
        self.z_overlay_label.setAlignment(Qt.AlignHCenter)
        self.z_overlay_high = QLabel('')
        self.z_overlay_low = QLabel('')
        self.x_overlay_label = QLabel('X  横向')
        self.x_overlay_label.setObjectName('eyebrow')
        self.x_overlay_label.setAlignment(Qt.AlignHCenter)
        self.x_overlay_high = QLabel('')
        self.x_overlay_low = QLabel('')
        for label in (self.z_overlay_high, self.z_overlay_low, self.x_overlay_high, self.x_overlay_low):
            label.setStyleSheet('color:#9d3b6c;font-size:10px;font-weight:650;background:transparent')
            label.setAlignment(Qt.AlignHCenter)
        z_group = QVBoxLayout()
        z_group.setContentsMargins(0, 0, 0, 0)
        z_group.setSpacing(1)
        for widget in (self.z_overlay_label, self.z_overlay_high, self.z_range_slider, self.z_overlay_low):
            z_group.addWidget(widget, 0 if widget is not self.z_range_slider else 1, Qt.AlignHCenter)
        x_group = QVBoxLayout()
        x_group.setContentsMargins(0, 0, 0, 0)
        x_group.setSpacing(1)
        for widget in (self.x_overlay_label, self.x_overlay_high, self.x_range_slider, self.x_overlay_low):
            x_group.addWidget(widget, 0 if widget is not self.x_range_slider else 1, Qt.AlignHCenter)
        z_widget, x_widget = QWidget(), QWidget()
        z_widget.setStyleSheet('background:#fff8fd')
        x_widget.setStyleSheet('background:#fff8fd')
        z_widget.setLayout(z_group)
        x_widget.setLayout(x_group)

        y_group = QVBoxLayout()
        y_group.setContentsMargins(0, 0, 0, 0)
        y_group.setSpacing(0)
        y_label = QLabel('Y  前后范围')
        y_label.setObjectName('eyebrow')
        y_label.setAlignment(Qt.AlignHCenter)
        y_ends = QHBoxLayout()
        self.y_overlay_low = QLabel('')
        self.y_overlay_high = QLabel('')
        for label in (self.y_overlay_low, self.y_overlay_high):
            label.setStyleSheet('color:#9d3b6c;font-size:10px;font-weight:650;background:transparent')
        y_ends.addWidget(self.y_overlay_low)
        y_ends.addStretch()
        y_ends.addWidget(self.y_overlay_high)
        y_group.addWidget(y_label)
        y_group.addLayout(y_ends)
        y_group.addWidget(self.y_range_slider)
        y_widget = QWidget()
        y_widget.setStyleSheet('background:#fff8fd')
        y_widget.setLayout(y_group)

        axes_row = QHBoxLayout()
        axes_row.setContentsMargins(0, 0, 0, 0)
        axes_row.setSpacing(0)
        z_widget.setFixedWidth(36)
        x_widget.setFixedWidth(36)
        axes_row.addWidget(z_widget)
        axes_row.addWidget(self.vtk_widget, 1)
        axes_row.addWidget(x_widget)
        vl.addLayout(axes_row, 1)
        y_row = QHBoxLayout()
        y_row.setContentsMargins(12, 0, 12, 0)
        y_row.addStretch()
        y_row.addWidget(y_widget)
        y_row.addStretch()
        vl.addLayout(y_row)
        ml.addWidget(view, 1)

        filter_panel = QFrame(objectName='filterpanel')
        filter_panel.setFixedHeight(78)
        fp = QHBoxLayout(filter_panel)
        fp.setContentsMargins(14, 8, 16, 7)
        fp.setSpacing(14)
        reset_button = self._button('重置', 'flat', self.reset_range, fp)
        reset_button.setFixedWidth(48)
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet('color:#e8ece9')
        fp.addWidget(separator)
        range_col = QVBoxLayout()
        range_col.setSpacing(2)
        range_title = QLabel('显示高度区间')
        range_title.setObjectName('muted')
        range_col.addWidget(range_title)
        fields = QHBoxLayout()
        fields.setSpacing(5)
        self.low_field = QLabel('−1.2 m')
        self.high_field = QLabel('8.4 m')
        for w in (self.low_field, self.high_field):
            w.setStyleSheet('background:transparent;border:none;padding:2px 0;color:#31483e;font-weight:600')
        fields.addWidget(QLabel('最小'))
        fields.addWidget(self.low_field)
        fields.addWidget(QLabel('至'))
        fields.addWidget(self.high_field)
        fields.addWidget(QLabel('最大'))
        fields.addStretch()
        range_col.addLayout(fields)
        legend = QHBoxLayout()
        grad = QLabel(' ')
        grad.setFixedHeight(4)
        grad.setStyleSheet('background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #2675a7,stop:.3 #39a88e,stop:.62 #b7c947,stop:1 #ed8c45);border-radius:4px')
        legend.addWidget(grad, 1)
        range_col.addLayout(legend)
        fp.addLayout(range_col, 1)
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setStyleSheet('color:#e8ece9')
        fp.addWidget(separator2)
        stats = QVBoxLayout()
        stats.setSpacing(1)
        stats.addWidget(QLabel('当前可见点数'))
        self.visible_count = QLabel('84,216')
        self.visible_count.setObjectName('metricValue')
        stats.addWidget(self.visible_count)
        self.visible_percent = QLabel('100% 点云')
        self.visible_percent.setObjectName('muted')
        stats.addWidget(self.visible_percent)
        fp.addLayout(stats)
        ml.addWidget(filter_panel)
        body.addWidget(main, 1)
        body.addWidget(self._build_filter_sidebar())
        root_layout.addLayout(body, 1)

        status = QFrame(objectName='toolbar')
        status.setFixedHeight(28)
        st = QHBoxLayout(status)
        st.setContentsMargins(15, 0, 13, 0)
        ready = QLabel('●  就绪')
        ready.setStyleSheet('color:#378461;font-size:10px')
        st.addWidget(ready)
        st.addSpacing(18)
        coords = QLabel('坐标系  XYZ · 米')
        coords.setObjectName('muted')
        st.addWidget(coords)
        st.addStretch()
        size_label = QLabel('点大小')
        size_label.setObjectName('muted')
        st.addWidget(size_label)
        self.size_slider = DragOnlySlider(Qt.Horizontal)
        self.size_slider.setRange(1, 8)
        self.size_slider.setValue(3)
        self.size_slider.setFixedWidth(95)
        self.size_slider.valueChanged.connect(self.set_point_size)
        st.addWidget(self.size_slider)
        self.size_text = QLabel('3 px')
        self.size_text.setObjectName('muted')
        st.addWidget(self.size_text)
        root_layout.addWidget(status)

        open_action = QAction(self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file)
        self.addAction(open_action)

    def _build_filter_sidebar(self):
        panel = TaffyPatternFrame(objectName='rightpanel')
        panel.setFixedWidth(286)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 12)
        layout.setSpacing(7)

        title_row = QHBoxLayout()
        title = QLabel('点云处理')
        title.setObjectName('sectionTitle')
        title_row.addWidget(title)
        title_row.addStretch()
        state = QLabel('处理工具')
        state.setObjectName('muted')
        title_row.addWidget(state)
        layout.addLayout(title_row)
        note = QLabel('选择一种处理方式')
        note.setObjectName('muted')
        layout.addWidget(note)

        process_layout = QVBoxLayout()
        process_layout.setContentsMargins(0, 0, 0, 0)
        process_layout.setSpacing(5)
        self.filter_expander = QToolButton()
        self.filter_expander.setObjectName('processHeader')
        self.filter_expander.setText('滤波')
        self.filter_expander.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.filter_expander.setArrowType(Qt.RightArrow)
        self.filter_expander.setCheckable(True)
        self.filter_expander.setChecked(False)
        self.filter_expander.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        process_layout.addWidget(self.filter_expander, 0, Qt.AlignTop)
        self.region_expander = QToolButton()
        self.region_expander.setObjectName('processHeader')
        self.region_expander.setText('删除与补全点云')
        self.region_expander.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.region_expander.setArrowType(Qt.RightArrow)
        self.region_expander.setCheckable(True)
        self.region_expander.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        process_layout.addWidget(self.region_expander, 0, Qt.AlignTop)
        layout.addLayout(process_layout, 0)

        content = QWidget()
        content.setStyleSheet('background:transparent')
        controls = QVBoxLayout(content)
        controls.setContentsMargins(8, 8, 8, 8)
        controls.setSpacing(7)
        self.filter_parameter_sliders = {}
        self.filter_enabled = QCheckBox('启用滤波')
        self.filter_enabled.setChecked(False)
        self.filter_enabled.toggled.connect(self.set_display_filter_enabled)
        controls.addWidget(self.filter_enabled)

        voxel_title = QHBoxLayout()
        voxel_name = QLabel('体素尺寸')
        voxel_name.setObjectName('muted')
        voxel_title.addWidget(voxel_name)
        voxel_title.addStretch()
        self.voxel_value = QLabel('0.02 m')
        self.voxel_value.setStyleSheet('background:transparent;color:#28785b;font-weight:650')
        voxel_title.addWidget(self.voxel_value)
        controls.addLayout(voxel_title)
        self.voxel_slider = DragOnlySlider(Qt.Horizontal)
        self.voxel_slider.setRange(1, 15)
        self.voxel_slider.setValue(2)
        self.voxel_slider.setTickPosition(QSlider.TicksBelow)
        self.voxel_slider.setTickInterval(1)
        self.voxel_slider.valueChanged.connect(self.schedule_display_filter)
        controls.addWidget(self.voxel_slider)
        voxel_limits = QHBoxLayout()
        low = QLabel('0.01 m')
        high = QLabel('0.15 m')
        for label in (low, high):
            label.setObjectName('muted')
        voxel_limits.addWidget(low)
        voxel_limits.addStretch()
        voxel_limits.addWidget(high)
        controls.addLayout(voxel_limits)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet('color:#e8ece9')
        controls.addSpacing(7)
        controls.addWidget(divider)
        controls.addSpacing(3)

        advanced = QLabel('高级清理')
        advanced.setObjectName('eyebrow')
        controls.addWidget(advanced)
        self.auto_density = QCheckBox('自动密度估计')
        self.auto_density.setChecked(False)
        self.auto_density.toggled.connect(self.schedule_display_filter)
        controls.addWidget(self.auto_density)

        mode_label = QLabel('自动滤波模式')
        mode_label.setObjectName('muted')
        controls.addWidget(mode_label)
        self.auto_mode = QComboBox()
        self.auto_mode.addItems(['detail  ·  温和', 'balanced  ·  均衡', 'coarse  ·  强力'])
        self.auto_mode.setStyleSheet('QComboBox { background:transparent;border:none;color:#40514b;padding:2px 0; } QComboBox::drop-down { border:none; }')
        self.auto_mode.currentIndexChanged.connect(self.schedule_display_filter)
        controls.addWidget(self.auto_mode)
        self._add_filter_parameter(controls, '自动采样点数', 'auto_sample', 100, 10000, 1000, '万点', 1)
        self._add_filter_parameter(controls, '体素下限', 'auto_min_voxel', 1, 30, 4, 'cm', 1)
        self._add_filter_parameter(controls, '体素上限', 'auto_max_voxel', 1, 30, 15, 'cm', 1)
        self._add_filter_parameter(controls, '半径下限', 'auto_min_radius', 5, 100, 12, 'cm', 1)
        self._add_filter_parameter(controls, '半径上限', 'auto_max_radius', 5, 100, 45, 'cm', 1)

        self.statistical_filter = QCheckBox('统计离群点滤波')
        self.statistical_filter.setChecked(True)
        self.statistical_filter.toggled.connect(self.schedule_display_filter)
        controls.addWidget(self.statistical_filter)
        self._add_filter_parameter(controls, '近邻数量', 'stat_neighbors', 2, 64, 8, '点', 1)
        self._add_filter_parameter(controls, '标准差阈值', 'stat_std_ratio', 10, 100, 35, 'σ', 10)

        self.radius_filter = QCheckBox('半径离群点滤波')
        self.radius_filter.setChecked(True)
        self.radius_filter.toggled.connect(self.schedule_display_filter)
        controls.addWidget(self.radius_filter)
        self._add_filter_parameter(controls, '半径内最少点数', 'radius_neighbors', 1, 100, 20, '点', 1)
        self._add_filter_parameter(controls, '搜索半径', 'radius_size', 2, 100, 5, 'cm', 1)

        self.write_filtered = QCheckBox('导出滤波后 PCD')
        self.write_filtered.setChecked(False)
        self.write_filtered.setToolTip('点击重新计算后，可另存当前滤波显示副本。')
        controls.addWidget(self.write_filtered)
        self.filter_content = content
        content.setVisible(False)
        self.filter_expander.toggled.connect(self._toggle_filter_expanded)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet('QScrollArea { background:#fffafd;border:1px solid #eadde8;border-radius:9px; } QWidget#qt_scrollarea_viewport { background:transparent; }')
        scroll.setWidget(content)
        process_layout.addWidget(scroll, 1)
        self.filter_scroll = scroll
        self.filter_status = QLabel('准备就绪')
        self.filter_status.setObjectName('muted')
        self.filter_status.setWordWrap(True)
        process_layout.addWidget(self.filter_status)
        self.render_count = QLabel('')
        self.render_count.setStyleSheet('background:transparent;color:#40584e;font-size:11px;font-weight:600')
        self.render_count.setWordWrap(True)
        process_layout.addWidget(self.render_count)
        self.filter_action_buttons = QWidget()
        action_layout = QVBoxLayout(self.filter_action_buttons)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        self._button('重新计算滤波', 'primary', self.recalculate_display_filter, action_layout)
        self._button('恢复未滤波显示', 'flat', self.restore_display_points, action_layout)
        process_layout.addWidget(self.filter_action_buttons)

        self.region_content = QWidget()
        region_controls = QVBoxLayout(self.region_content)
        region_controls.setContentsMargins(8, 8, 8, 8)
        region_controls.setSpacing(7)
        region_help = QLabel('双击选择至少 4 个点，计算凸包后，再选择删除或补全该立体区域。')
        region_help.setObjectName('muted')
        region_help.setWordWrap(True)
        region_controls.addWidget(region_help)
        self._button('开始选点区域', 'flat', self.start_region_selection, region_controls)
        self.region_pick_status = QLabel('尚未选择边界点')
        self.region_pick_status.setObjectName('muted')
        self.region_pick_status.setWordWrap(True)
        region_controls.addWidget(self.region_pick_status)
        self._add_filter_parameter(region_controls, '区域外扩（XYZ）', 'region_expand', 0, 100, 5, 'cm', 1)
        self._button('计算凸包区域', 'flat', self.calculate_selected_region, region_controls)
        self.delete_region_button = self._button('删除区域点', 'primary', self.delete_selected_region, region_controls)
        divider2 = QFrame()
        divider2.setFrameShape(QFrame.HLine)
        divider2.setStyleSheet('color:#e8ece9')
        region_controls.addWidget(divider2)
        self.complete_region_button = self._button('补全区域点', 'flat', self.complete_selected_region, region_controls)
        self._button('清空点', 'flat', self.clear_completion_points, region_controls)
        self._add_filter_parameter(region_controls, '补全点间距', 'completion_spacing', 2, 50, 10, 'm', 100)
        self._button('取消点选模式', 'flat', self.cancel_point_edit_mode, region_controls)
        region_scroll = QScrollArea()
        region_scroll.setWidgetResizable(True)
        region_scroll.setFrameShape(QFrame.NoFrame)
        region_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        region_scroll.setStyleSheet('QScrollArea { background:#fffafd;border:1px solid #eadde8;border-radius:9px; } QWidget#qt_scrollarea_viewport { background:transparent; }')
        region_scroll.setWidget(self.region_content)
        process_layout.addWidget(region_scroll, 1)
        self.region_scroll = region_scroll
        self.region_expander.toggled.connect(self._toggle_region_expanded)
        self.region_scroll.setVisible(False)
        layout.addStretch(1)
        self.filter_status.setVisible(False)
        self.render_count.setVisible(False)
        self.filter_action_buttons.setVisible(False)
        self.filter_scroll.setVisible(False)
        return panel

    def _toggle_filter_expanded(self, expanded):
        self.filter_expander.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.filter_content.setVisible(expanded)
        self.filter_scroll.setVisible(expanded)
        self.filter_status.setVisible(expanded)
        self.render_count.setVisible(expanded)
        self.filter_action_buttons.setVisible(expanded)

    def _toggle_region_expanded(self, expanded):
        self.region_expander.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.region_scroll.setVisible(expanded)
        if not expanded:
            self.cancel_point_edit_mode()

    def start_region_selection(self):
        self.selected_hull_equations = None
        self.selected_region_vertices = None
        self.region_pick_scope = dict(self.axis_ranges)
        self._clear_region_preview()
        self.region_pick_points.clear()
        self._clear_region_markers()
        self.point_edit_mode = 'delete'
        self._set_point_pick_mode(True)
        self.region_pick_status.setText('双击选择区域顶点，至少 4 个不共面的点；然后点击“计算凸包区域”。')

    def finish_region_selection(self):
        if self.point_edit_mode == 'delete':
            self.point_edit_mode = None
            self._set_point_pick_mode(False)
            self.region_pick_status.setText(f'已选 {len(self.region_pick_points)} 个点；点击“计算凸包区域”。')

    def calculate_selected_region(self):
        selected = np.asarray(self.region_pick_points, dtype=np.float64)
        if len(np.unique(selected, axis=0)) < 4:
            self.region_pick_status.setText('至少双击选择 4 个不同的点。')
            return
        expand_m = self.filter_parameter_sliders['region_expand'][0].value() / 100
        try:
            # Expand the selected geometry by at most ``expand_m`` on each
            # coordinate axis, then recompute the convex hull.  Moving the
            # original facet planes along their normals can amplify distances
            # dramatically at sharp corners, which made a 5 cm setting look
            # like metres.  The 8 offsets below form the XYZ Minkowski box
            # dilation and keep the expansion bounded and predictable.
            offsets = np.array([(sx, sy, sz)
                                for sx in (-expand_m, expand_m)
                                for sy in (-expand_m, expand_m)
                                for sz in (-expand_m, expand_m)], dtype=np.float64)
            expanded_samples = (selected[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
            expanded_hull = ConvexHull(expanded_samples)
            self.selected_hull_equations = expanded_hull.equations.copy()
            self.selected_region_vertices = expanded_hull.points[expanded_hull.vertices]
            preview_hull = expanded_hull
        except QhullError:
            self.region_pick_status.setText('所选点无法形成有效三维凸包，请选择至少 4 个不共面的点。')
            self.selected_hull_equations = None
            return
        self._show_region_preview(preview_hull)
        self.point_edit_mode = None
        self._set_point_pick_mode(False)
        self.region_pick_status.setText(f'区域已计算：{len(selected)} 个点，XYZ 各方向外扩 {expand_m * 100:.0f} cm。选择删除或补全。')

    def _show_region_preview(self, hull):
        self._clear_region_preview()
        points = vtk.vtkPoints()
        for point in hull.points:
            points.InsertNextPoint(*map(float, point))
        faces = vtk.vtkCellArray()
        for triangle in hull.simplices:
            faces.InsertNextCell(3)
            for point_id in triangle:
                faces.InsertCellPoint(int(point_id))
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(faces)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.91, 0.23, 0.48)
        actor.GetProperty().SetOpacity(0.22)
        actor.GetProperty().SetEdgeVisibility(True)
        actor.GetProperty().SetEdgeColor(0.67, 0.20, 0.44)
        actor.GetProperty().SetLineWidth(1.5)
        self.renderer.AddActor(actor)
        self.region_preview_actor = actor
        self.vtk_widget.GetRenderWindow().Render()

    def _clear_region_preview(self):
        if self.region_preview_actor is not None:
            self.renderer.RemoveActor(self.region_preview_actor)
            self.region_preview_actor = None

    def _get_full_edit_points(self):
        layer = self._find_layer(self.active_layer_id)
        if layer is None:
            return np.empty((0, 3), dtype=np.float32)
        if 'edit_points' in layer:
            return layer['edit_points']
        if len(layer['source_points']) >= layer['total_points']:
            layer['edit_points'] = layer['source_points']
            return layer['edit_points']
        source_path = layer.get('source_path')
        if not source_path or not Path(source_path).is_file():
            QMessageBox.warning(self, '无法载入完整点云', '原始 PCD 文件不可用，当前只能显示抽样点，无法安全编辑。')
            return layer['source_points']
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            points = load_full_xyz(source_path)
            points = points[np.isfinite(points).all(axis=1)]
            layer['edit_points'] = np.ascontiguousarray(points, dtype=np.float32)
            self.region_pick_status.setText(f'已载入完整点云：{fmt_number(len(points))} 点。')
            return layer['edit_points']
        except Exception as error:
            QMessageBox.warning(self, '完整点云载入失败', str(error))
            return layer['source_points']
        finally:
            QApplication.restoreOverrideCursor()

    def clear_completion_points(self):
        if self.completion_base_points is None:
            self.region_pick_status.setText('当前没有本轮放置的补全点。')
            return
        self.point_edit_mode = None
        self._set_point_pick_mode(False)
        self._replace_active_source_points(self.completion_base_points)
        self.completion_base_points = None
        self.completion_markers.clear()
        self.region_pick_status.setText('已清空本轮放置的补全点。')

    def cancel_point_edit_mode(self):
        self.point_edit_mode = None
        self.region_pick_scope = None
        self._set_point_pick_mode(False)
        self._clear_region_markers()
        self.region_pick_status.setText('点选模式已取消。')

    def _set_point_pick_mode(self, enabled):
        self.vtk_widget.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def _clear_region_markers(self):
        for actor in self.region_marker_actors:
            self.renderer.RemoveActor(actor)
        self.region_marker_actors.clear()
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.GetRenderWindow().Render()

    def _on_point_pick(self, _caller, _event):
        return

    def _on_embedded_double_click(self, x, y):
        if self.point_edit_mode == 'delete':
            self._pick_delete_at_screen(x, y)

    def _add_region_marker(self, point):
        source = vtk.vtkSphereSource()
        source.SetCenter(*map(float, point))
        source.SetRadius(max(0.025, (self.z_max - self.z_min) * 0.002))
        source.SetThetaResolution(10)
        source.SetPhiResolution(8)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.91, 0.23, 0.48)
        self.renderer.AddActor(actor)
        self.region_marker_actors.append(actor)
        self.vtk_widget.GetRenderWindow().Render()

    def _picked_render_point_id(self, x, y):
        # The picker indexes the polydata currently attached to the mapper.
        # That is ``self.points`` (possibly filtered/sampled), not the source
        # cloud.  Using render_source_points here makes the index refer to a
        # different point as soon as filtering changes the displayed array.
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.02)
        picker.PickFromListOn()
        picker.AddPickList(self.points_actor)
        if not picker.Pick(x, y, 0, self.renderer):
            return -1
        actor = picker.GetActor()
        if actor is None or actor.GetMapper() is None:
            return -1
        if actor.GetMapper().GetAddressAsString('') != self.mapper.GetAddressAsString(''):
            return -1
        point_id = picker.GetPointId()
        return point_id if self.points is not None and 0 <= point_id < len(self.points) else -1

    def _picked_visible_point(self, x, y):
        point_id = self._picked_render_point_id(x, y)
        if point_id < 0:
            return None
        point = self.points[point_id].copy()
        # VTK clipping is a rendering concern; explicitly enforce the same
        # current XYZ window for editing so hidden points can never be picked.
        scope = self.region_pick_scope or self.axis_ranges
        if not all(scope[axis][0] <= point[i] <= scope[axis][1]
                   for i, axis in enumerate(('x', 'y', 'z'))):
            return None
        return point

    def _pick_delete_at_screen(self, x, y):
        if self.point_edit_mode != 'delete':
            return
        point = self._picked_visible_point(x, y)
        if point is None:
            return
        self.region_pick_points.append(point)
        self._add_region_marker(point)
        self.selected_hull_equations = None
        self.region_pick_status.setText(f'已选 {len(self.region_pick_points)} 个区域顶点；点击“计算凸包区域”。')

    def _pick_completion_at_screen(self, x, y):
        if self.point_edit_mode != 'completion':
            return
        point = self._picked_visible_point(x, y)
        if point is None:
            self.region_pick_status.setText('没有命中点云，请双击点云点位。')
            return
        self.complete_cloud_at(point)

    def _replace_active_source_points(self, points):
        points = np.ascontiguousarray(points, dtype=np.float32)
        layer = self._find_layer(self.active_layer_id)
        if layer is None or not len(points):
            return
        render_stride = max(1, int(np.ceil(len(points) / MAX_RENDER_POINTS)))
        display_points = points[::render_stride].copy()
        layer['source_points'] = display_points
        layer['edit_points'] = points
        layer['display_points'] = display_points.copy()
        layer['total_points'] = len(points)
        layer['full_z'] = points[:, 2].copy()
        layer['axis_bounds'] = {axis: (float(points[:, i].min()), float(points[:, i].max())) for i, axis in enumerate('xyz')}
        layer['axis_ranges'] = dict(layer['axis_bounds'])
        self.render_source_points = display_points
        self.full_z, self.total_points = layer['full_z'], layer['total_points']
        self.axis_bounds, self.axis_ranges = dict(layer['axis_bounds']), dict(layer['axis_ranges'])
        self.filter_request_id += 1
        self.filter_enabled.setChecked(False)
        self._set_slider_extent()
        self._set_render_points(display_points)
        self._update_visible_stats()
        self.render_count.setText(f'渲染 {fmt_number(len(points))} 点')
        self.vtk_widget.GetRenderWindow().Render()

    def delete_selected_region(self):
        if self.selected_hull_equations is None:
            QMessageBox.warning(self, '区域尚未计算', '请先选至少 4 个不同且不共面的点，再点击“计算凸包区域”。')
            return
        if len(self.region_pick_points) < 4:
            QMessageBox.warning(self, '边界点不足', '删除区域至少需要 4 个点，请继续选点。')
            return
        layer = self._find_layer(self.active_layer_id)
        if layer is None:
            return
        if 'edit_points' in layer:
            self._delete_region_from_points(layer['edit_points'], visible_only=True)
            return
        if len(layer['source_points']) >= layer['total_points']:
            layer['edit_points'] = layer['source_points']
            self._delete_region_from_points(layer['edit_points'], visible_only=True)
            return
        source_path = layer.get('source_path')
        if not source_path or not Path(source_path).is_file():
            QMessageBox.warning(self, '无法载入完整点云', '原始 PCD 文件不可用，无法安全删除抽样之外的点。')
            return
        if self.edit_load_worker is not None and self.edit_load_worker.isRunning():
            return
        self.edit_load_request += 1
        request_id = self.edit_load_request
        self.edit_load_layer_id = layer['id']
        self.region_pick_status.setText('正在后台读取完整点云以执行删除…')
        self.delete_region_button.setEnabled(False)
        worker = FullPointLoadWorker(request_id, source_path)
        worker.loaded.connect(self._finish_full_point_load_for_delete)
        worker.failed.connect(self._fail_full_point_load_for_delete)
        worker.finished.connect(worker.deleteLater)
        self.edit_load_worker = worker
        worker.start()

    def complete_selected_region(self):
        if self.selected_hull_equations is None:
            QMessageBox.warning(self, '区域尚未计算', '请先选至少 4 个不同且不共面的点，再点击“计算凸包区域”。')
            return
        layer = self._find_layer(self.active_layer_id)
        if layer is None:
            return
        points = layer.get('edit_points', layer['source_points'])
        visible_mask = np.ones(len(points), dtype=bool)
        for column, axis in enumerate(('x', 'y', 'z')):
            visible_mask &= (points[:, column] >= self.axis_ranges[axis][0]) & (points[:, column] <= self.axis_ranges[axis][1])
        visible_points = points[visible_mask]
        if len(visible_points) < 6:
            self.region_pick_status.setText('可用点太少，无法补全区域。')
            return
        equations = self.selected_hull_equations
        region_vertices = self.selected_region_vertices
        if region_vertices is None or not len(region_vertices):
            self.region_pick_status.setText('凸包区域顶点无效，请重新计算凸包。')
            return
        low = region_vertices.min(axis=0)
        high = region_vertices.max(axis=0)
        visible_low = np.array([self.axis_ranges[axis][0] for axis in ('x', 'y', 'z')], dtype=np.float64)
        visible_high = np.array([self.axis_ranges[axis][1] for axis in ('x', 'y', 'z')], dtype=np.float64)
        low = np.maximum(low, visible_low)
        high = np.minimum(high, visible_high)
        if np.any(high <= low):
            self.region_pick_status.setText('外扩区域与当前可视范围没有交集。')
            return
        spacing = self.filter_parameter_sliders['completion_spacing'][0].value() / 100
        # Completion is intentionally approximate: sample random points in
        # the expanded hull's bounding box instead of building a full grid.
        # The old grid grew cubically with the region and could freeze the UI.
        volume = float(np.prod(np.maximum(high - low, 0.0)))
        target_count = int(np.clip(volume / max(spacing ** 3, 1e-9) * 0.6, 200, 80_000))
        rng = np.random.default_rng()
        candidates = np.empty((0, 3), dtype=np.float32)
        remaining = target_count
        for _ in range(8):
            if remaining <= 0:
                break
            batch_size = min(remaining * 3, 60_000)
            batch = rng.uniform(low, high, size=(batch_size, 3)).astype(np.float32)
            inside = np.all(batch @ equations[:, :3].T + equations[:, 3] <= 1e-6, axis=1)
            accepted = batch[inside]
            if len(accepted):
                take = min(remaining, len(accepted))
                candidates = np.concatenate((candidates, accepted[:take]))
                remaining -= take
        if not len(candidates):
            self.region_pick_status.setText('随机采样没有命中凸包区域，请重新计算。')
            return
        self._replace_active_source_points(np.concatenate((points, candidates)))
        self.region_pick_status.setText(f'已补全区域内 {fmt_number(len(candidates))} 个点，间距约 {spacing:.2f} m。')

    def _finish_full_point_load_for_delete(self, request_id, points):
        self.edit_load_worker = None
        self.delete_region_button.setEnabled(True)
        if request_id != self.edit_load_request or self.edit_load_layer_id != self.active_layer_id:
            return
        layer = self._find_layer(self.active_layer_id)
        if layer is None:
            return
        layer['edit_points'] = points
        self.region_pick_status.setText(f'完整点云已载入：{fmt_number(len(points))} 点，正在删除…')
        self._delete_region_from_points(points, visible_only=True)

    def _fail_full_point_load_for_delete(self, request_id, error):
        self.edit_load_worker = None
        self.delete_region_button.setEnabled(True)
        if request_id == self.edit_load_request:
            self.region_pick_status.setText(f'完整点云读取失败：{error}')
            QMessageBox.warning(self, '完整点云读取失败', error)

    def _delete_region_from_points(self, points, visible_only=False):
        expand_m = self.filter_parameter_sliders['region_expand'][0].value() / 100
        equations = self.selected_hull_equations
        if equations is None:
            return
        remove = np.zeros(len(points), dtype=bool)
        visible_mask = np.ones(len(points), dtype=bool)
        if visible_only:
            for column, axis in enumerate(('x', 'y', 'z')):
                visible_mask &= (points[:, column] >= self.axis_ranges[axis][0]) & (points[:, column] <= self.axis_ranges[axis][1])
        for start in range(0, len(points), 250_000):
            chunk = points[start:start + 250_000]
            inside = np.all(chunk @ equations[:, :3].T + equations[:, 3] <= 1e-6, axis=1)
            remove[start:start + len(chunk)] = inside & visible_mask[start:start + len(chunk)]
        removed = int(remove.sum())
        if not removed:
            QMessageBox.information(self, '没有匹配点', '所选立体区域内没有可删除的点。')
            return
        if removed == len(points):
            QMessageBox.warning(self, '无法删除', '该区域包含图层全部点，至少保留一个点。')
            return
        self._replace_active_source_points(points[~remove])
        if self.completion_base_points is not None:
            self.completion_base_points = self._remove_points_in_hull(self.completion_base_points, equations)
        self.completion_markers = [
            self._remove_points_in_hull(batch, equations)
            for batch in self.completion_markers
        ]
        self._clear_region_markers()
        self.region_pick_status.setText(f'已删除 {fmt_number(removed)} 个点（XYZ 各方向外扩 {expand_m * 100:.0f} cm）。')

    @staticmethod
    def _remove_points_in_hull(points, equations):
        remove = np.zeros(len(points), dtype=bool)
        for start in range(0, len(points), 250_000):
            chunk = points[start:start + 250_000]
            inside = np.all(chunk @ equations[:, :3].T + equations[:, 3] <= 1e-6, axis=1)
            remove[start:start + len(chunk)] = inside
        return points[~remove].copy()

    def complete_cloud_at(self, center):
        source_for_fit = self.completion_base_points if self.completion_base_points is not None else self.render_source_points
        points = source_for_fit
        visible = np.ones(len(points), dtype=bool)
        for axis, column in (('x', 0), ('y', 1), ('z', 2)):
            visible &= (points[:, column] >= self.axis_ranges[axis][0]) & (points[:, column] <= self.axis_ranges[axis][1])
        points = points[visible]
        if len(points) < 6:
            QMessageBox.warning(self, '补全失败', '可见点太少，无法拟合局部表面。')
            return
        local_distance2 = np.sum((points - center) ** 2, axis=1)
        candidate_count = min(48, len(points))
        indices = np.argpartition(local_distance2, candidate_count - 1)[:candidate_count]
        distances = np.sqrt(local_distance2[indices])
        neighbors = points[indices]
        origin = neighbors.mean(axis=0)
        _, _, basis = np.linalg.svd(neighbors - origin, full_matrices=False)
        normal, tangent_u, tangent_v = basis[-1], basis[0], basis[1]
        spacing = self.filter_parameter_sliders['completion_spacing'][0].value() / 100
        radius = min(max(spacing * 4, float(np.max(distances)) * 1.5), 0.75)
        axis = np.arange(-radius, radius + spacing * 0.5, spacing)
        uu, vv = np.meshgrid(axis, axis)
        disk = uu ** 2 + vv ** 2 <= radius ** 2
        generated = origin + uu[disk, None] * tangent_u + vv[disk, None] * tangent_v
        generated -= ((generated - origin) @ normal)[:, None] * normal
        generated = generated.astype(np.float32)
        if len(generated) > 4000:
            stride = int(np.ceil(len(generated) / 4000))
            generated = generated[::stride]
        current_points = self.render_source_points
        local = current_points[np.sum((current_points - center) ** 2, axis=1) <= (radius + spacing) ** 2]
        if len(local):
            nearest_distance2 = np.min(np.sum((generated[:, None, :] - local[None, :, :]) ** 2, axis=2), axis=1)
            generated = generated[nearest_distance2 > (spacing * 0.2) ** 2]
        if not len(generated):
            self.region_pick_status.setText('这个位置已有同样密度的点；调整“补全点间距”后再双击。')
            return
        self._replace_active_source_points(np.concatenate((current_points, generated)))
        self.completion_markers.append(generated.copy())
        self.point_edit_mode = 'completion'
        self._set_point_pick_mode(True)
        self.region_pick_status.setText(f'已补全 {fmt_number(len(generated))} 个点；累计 {fmt_number(sum(len(part) for part in self.completion_markers))} 个，间距约 {spacing:.2f} m。')

    def _add_filter_parameter(self, layout, title, key, minimum, maximum, default, unit, multiplier):
        row = QHBoxLayout()
        label = QLabel(title)
        label.setObjectName('muted')
        value = QLabel()
        value.setStyleSheet('color:#40584e;background:transparent')
        row.addWidget(label)
        row.addStretch()
        row.addWidget(value)
        layout.addLayout(row)
        slider = DragOnlySlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(default)
        slider.valueChanged.connect(lambda value, parameter=key: self._update_filter_parameter(parameter, value))
        slider.valueChanged.connect(self.schedule_display_filter)
        layout.addWidget(slider)
        self.filter_parameter_sliders[key] = (slider, value, unit, multiplier)
        self._update_filter_parameter(key, default)

    def _update_filter_parameter(self, key, raw_value):
        slider, label, unit, multiplier = self.filter_parameter_sliders[key]
        value = raw_value / multiplier
        label.setText(f'{value:g} {unit}')

    def _button(self, text, name, callback, layout):
        button = QPushButton(text)
        button.setObjectName(name)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _hold_rotate_button(self, symbol, tooltip, direction, layout):
        button = QPushButton(symbol)
        button.setObjectName('flat')
        button.setFixedWidth(30)
        button.setToolTip(tooltip)
        button.pressed.connect(lambda: self.start_held_rotation(direction))
        button.released.connect(self.stop_held_rotation)
        layout.addWidget(button)
        return button

    def effective_voxel_size(self, points):
        if not self.auto_density.isChecked() or not len(points):
            size = self.voxel_slider.value() / 100
            self.voxel_value.setText(f'{size:.2f} m')
            return size
        max_samples = self.filter_parameter_sliders['auto_sample'][0].value() * 10000
        stride = max(1, int(np.ceil(len(points) / max_samples)))
        sample = points[::stride]
        spans = np.ptp(sample, axis=0)
        spacing = float(np.cbrt(np.prod(np.maximum(spans, 1e-4)) / len(sample)))
        auto_min = self.filter_parameter_sliders['auto_min_voxel'][0].value() / 100
        auto_max = self.filter_parameter_sliders['auto_max_voxel'][0].value() / 100
        mode_scale = (1.25, 1.75, 2.5)[self.auto_mode.currentIndex()]
        size = float(np.clip(spacing * mode_scale, min(auto_min, auto_max), max(auto_min, auto_max)))
        self.voxel_value.setText(f'自动 {size:.2f} m')
        return size

    def current_visible_source_points(self):
        if self.render_source_points is None:
            return np.empty((0, 3), dtype=np.float32)
        visible = np.ones(len(self.render_source_points), dtype=bool)
        for axis, column in (('x', 0), ('y', 1), ('z', 2)):
            visible &= (self.render_source_points[:, column] >= self.axis_ranges[axis][0])
            visible &= (self.render_source_points[:, column] <= self.axis_ranges[axis][1])
        return self.render_source_points[visible]

    def set_display_filter_enabled(self, enabled):
        self.filter_request_id += 1
        if enabled:
            self.schedule_display_filter()
            return
        self.filter_debounce.stop()
        self._set_render_points(self.render_source_points)
        self.filter_status.setText('滤波已关闭，显示原始视图副本。')
        self.render_count.setText(f'渲染 {fmt_number(len(self.points))} 点')
        self.vtk_widget.GetRenderWindow().Render()

    def schedule_display_filter(self):
        if self.render_source_points is None:
            return
        self.filter_request_id += 1
        if not self.filter_enabled.isChecked():
            return
        visible_points = self.current_visible_source_points()
        self.effective_voxel_size(visible_points)
        self.filter_status.setText(f'参数已更改；点击重新计算处理当前可见 {fmt_number(len(visible_points))} 点。')

    def recalculate_display_filter(self):
        if not self.filter_enabled.isChecked():
            self.filter_status.setText('请先开启温和滤波。')
            return
        self.filter_request_id += 1
        self.filter_recalculate_requested = True
        self.filter_debounce.stop()
        self.start_display_filter()

    def start_display_filter(self):
        if self.render_source_points is None or not self.filter_enabled.isChecked():
            return
        if self.filter_worker is not None and self.filter_worker.isRunning():
            self.filter_status.setText('正在完成上一项设置…')
            return
        visible_points = self.current_visible_source_points()
        voxel_size = self.effective_voxel_size(visible_points)
        if not len(visible_points):
            self.filter_status.setText('当前高度范围没有可滤波的点。')
            return
        self.filter_status.setText(f'正在后台计算 {fmt_number(len(visible_points))} 个可见点…')
        radius_size = self.filter_parameter_sliders['radius_size'][0].value() / 100
        if self.auto_density.isChecked():
            max_samples = self.filter_parameter_sliders['auto_sample'][0].value() * 10000
            stride = max(1, int(np.ceil(len(visible_points) / max_samples)))
            sample = visible_points[::stride]
            spans = np.ptp(sample, axis=0)
            spacing = float(np.cbrt(np.prod(np.maximum(spans, 1e-4)) / len(sample)))
            radius_min = self.filter_parameter_sliders['auto_min_radius'][0].value() / 100
            radius_max = self.filter_parameter_sliders['auto_max_radius'][0].value() / 100
            mode_scale = (3.0, 4.0, 5.0)[self.auto_mode.currentIndex()]
            radius_size = float(np.clip(spacing * mode_scale, min(radius_min, radius_max), max(radius_min, radius_max)))
        worker = DisplayFilterWorker(
            self.filter_request_id,
            visible_points,
            voxel_size,
            self.statistical_filter.isChecked(),
            self.filter_parameter_sliders['stat_neighbors'][0].value(),
            self.filter_parameter_sliders['stat_std_ratio'][0].value() / 10,
            self.radius_filter.isChecked(),
            self.filter_parameter_sliders['radius_neighbors'][0].value(),
            radius_size,
        )
        self.filter_scope = dict(self.axis_ranges)
        worker.completed.connect(self.finish_display_filter)
        worker.failed.connect(self.fail_display_filter)
        worker.finished.connect(worker.deleteLater)
        self.filter_worker = worker
        self.filter_recalculate_requested = False
        worker.start()

    def finish_display_filter(self, request_id, points):
        self.filter_worker = None
        if request_id != self.filter_request_id:
            if self.filter_enabled.isChecked() and self.filter_recalculate_requested:
                QTimer.singleShot(0, self.start_display_filter)
            return
        scope = self.filter_scope or self.axis_ranges
        visible_mask = np.ones(len(self.render_source_points), dtype=bool)
        for column, axis in enumerate(('x', 'y', 'z')):
            low, high = scope[axis]
            visible_mask &= (self.render_source_points[:, column] >= low) & (self.render_source_points[:, column] <= high)
        outside = self.render_source_points[~visible_mask]
        self._set_render_points(np.concatenate((outside, points)))
        self.filter_status.setText('滤波结果已保留；计算范围外的点仍为原始点。')
        self.render_count.setText(f'当前范围处理 {fmt_number(len(points))} 点 · 渲染 {fmt_number(len(self.points))} 点')
        self.vtk_widget.GetRenderWindow().Render()
        if self.write_filtered.isChecked():
            path, _ = QFileDialog.getSaveFileName(self, '另存滤波点云', 'filtered_view.pcd', 'PCD 点云 (*.pcd)')
            if path:
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points)
                if o3d.io.write_point_cloud(path, cloud, write_ascii=False, compressed=False):
                    self.filter_status.setText(f'滤波结果已另存：{Path(path).name}')
                else:
                    QMessageBox.warning(self, '导出失败', '无法写入指定的 PCD 文件。')
        self.show_filter_result_popup(True)

    def fail_display_filter(self, request_id, error):
        self.filter_worker = None
        if request_id == self.filter_request_id:
            self.filter_status.setText(f'滤波未完成：{error}')
            self.show_filter_result_popup(False)

    def show_filter_result_popup(self, success):
        """Show a short, non-blocking result notice for the current filter job."""
        if self.filter_result_popup is not None:
            self.filter_result_popup.close()
        popup = QMessageBox(self)
        popup.setWindowTitle('滤波完成' if success else '滤波失败')
        popup.setText('滤波成功喵' if success else '滤波失败喵')
        popup.setIcon(QMessageBox.Information if success else QMessageBox.Warning)
        popup.setStandardButtons(QMessageBox.Ok)
        popup.setAttribute(Qt.WA_DeleteOnClose)
        self.filter_result_popup = popup
        popup.finished.connect(lambda _result: setattr(self, 'filter_result_popup', None))
        popup.show()
        close_timer = QTimer(popup)
        close_timer.setSingleShot(True)
        close_timer.timeout.connect(popup.accept)
        close_timer.start(1900)

    def restore_display_points(self):
        if self.filter_enabled.isChecked():
            self.filter_enabled.setChecked(False)
        else:
            self._set_render_points(self.render_source_points)
            self.vtk_widget.GetRenderWindow().Render()
        self.filter_status.setText('已恢复原始视图副本。')

    def _build_vtk(self):
        render_window = self.vtk_widget.GetRenderWindow()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.978, 0.949, 0.975)
        self.renderer.SetBackground2(0.948, 0.912, 0.958)
        self.renderer.GradientBackgroundOn()
        render_window.AddRenderer(self.renderer)
        self.interactor = render_window.GetInteractor()
        # Use conventional drag controls instead of VTK's joystick-style camera.
        # Left drag rotates, middle drag pans, and right drag zooms only while moving.
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.interactor.Initialize()
        self.points_actor = vtk.vtkActor()
        self.mapper = vtk.vtkPolyDataMapper()
        self.points_actor.SetMapper(self.mapper)
        self.renderer.AddActor(self.points_actor)
        self.low_plane = vtk.vtkPlane()
        self.high_plane = vtk.vtkPlane()
        self.x_low_plane = vtk.vtkPlane()
        self.x_high_plane = vtk.vtkPlane()
        self.y_low_plane = vtk.vtkPlane()
        self.y_high_plane = vtk.vtkPlane()
        self.mapper.AddClippingPlane(self.low_plane)
        self.mapper.AddClippingPlane(self.high_plane)
        self.mapper.AddClippingPlane(self.x_low_plane)
        self.mapper.AddClippingPlane(self.x_high_plane)
        self.mapper.AddClippingPlane(self.y_low_plane)
        self.mapper.AddClippingPlane(self.y_high_plane)
        self.mapper.SetScalarModeToUsePointFieldData()
        self.mapper.SelectColorArray('Elevation')
        self.mapper.SetScalarVisibility(True)
        self.lookup = vtk.vtkLookupTable()
        self.lookup.SetNumberOfTableValues(256)
        self.lookup.Build()
        stops = [(0.00, (0.13, .42, .65)), (.28, (.19, .67, .65)), (.57, (.68, .76, .31)), (.80, (.93, .69, .24)), (1.0, (.85, .36, .24))]
        for i in range(256):
            t = i / 255
            for k in range(len(stops) - 1):
                if stops[k][0] <= t <= stops[k + 1][0]:
                    a, ca = stops[k]
                    b, cb = stops[k + 1]
                    f = (t - a) / (b - a)
                    color = tuple(ca[j] * (1 - f) + cb[j] * f for j in range(3))
                    self.lookup.SetTableValue(i, *color, 1)
                    break
        self.mapper.SetLookupTable(self.lookup)
        self.grid_actor = self._make_grid()
        self.renderer.AddActor(self.grid_actor)
        self.renderer.SetUseDepthPeeling(True)

    def _make_grid(self):
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        extent, step = 50, 5
        for pos in range(-extent, extent + 1, step):
            for a, b in (((-extent, pos, -.45), (extent, pos, -.45)), ((pos, -extent, -.45), (pos, extent, -.45))):
                idx = points.GetNumberOfPoints()
                points.InsertNextPoint(*a)
                points.InsertNextPoint(*b)
                lines.InsertNextCell(2)
                lines.InsertCellPoint(idx)
                lines.InsertCellPoint(idx + 1)
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(.78, .69, .80)
        actor.GetProperty().SetOpacity(.58)
        actor.GetProperty().SetLineWidth(1)
        return actor

    def _add_layer_row(self, layer):
        layer_id = layer['id']
        row = QFrame()
        row.setObjectName('layerRow')
        row.setStyleSheet('QFrame#layerRow { background: transparent; border: 1px solid transparent; border-radius: 4px; } QFrame#layerRow[selected="true"] { background: #e8f3ed; border-color: #c8ded3; }')
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(4, 2, 3, 2)
        row_layout.setSpacing(3)
        eye = QToolButton()
        eye.setFixedSize(24, 26)
        eye.setStyleSheet('QToolButton { border: none; background: transparent; color: #28785b; font-size: 15px; } QToolButton:hover { background: #edf4f0; border-radius: 3px; }')
        eye.setToolTip('隐藏图层' if layer['visible'] else '显示图层')
        eye.setText('◉' if layer['visible'] else '○')
        eye.clicked.connect(lambda checked=False, target=layer_id: self.toggle_layer_visibility(target))
        select = LayerSelectButton(layer['filename'])
        select.setObjectName('layerSelect')
        select.setStyleSheet('QPushButton#layerSelect { text-align: left; border: none; background: transparent; padding: 5px 2px; color: #3f504a; } QPushButton#layerSelect:hover { color: #20785b; background: transparent; }')
        select.clicked.connect(lambda checked=False, target=layer_id: self.activate_layer(target))
        select.doubleClicked.connect(lambda target=layer_id: self.show_and_activate_layer(target))
        menu_button = QToolButton()
        menu_button.setText('•••')
        menu_button.setFixedSize(26, 26)
        menu_button.setToolTip('图层操作')
        menu_button.setStyleSheet('QToolButton { border: none; background: transparent; color: #74817c; padding: 2px; } QToolButton:hover { background: #edf4f0; border-radius: 3px; }')
        menu_button.clicked.connect(lambda checked=False, target=layer_id, button=menu_button: self.show_layer_menu(target, button))
        row_layout.addWidget(eye)
        row_layout.addWidget(select, 1)
        row_layout.addWidget(menu_button)
        details = QLabel(f"{fmt_number(layer['total_points'])} 点 · XYZ")
        details.setObjectName('muted')
        wrapper = QWidget()
        wrapper.setStyleSheet('background:transparent')
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 2)
        wrapper_layout.setSpacing(0)
        wrapper_layout.addWidget(row)
        wrapper_layout.addWidget(details)
        self.layers_layout.addWidget(wrapper)
        layer['row'] = row
        layer['eye'] = eye
        layer['select'] = select
        layer['menu_button'] = menu_button
        layer['details'] = details
        self.layer_rows[layer_id] = wrapper
        self._refresh_layer_selection()

    def _refresh_layer_selection(self):
        for layer in self.layers:
            row = layer.get('row')
            if row is None:
                continue
            row.setProperty('selected', layer['id'] == self.active_layer_id)
            row.style().unpolish(row)
            row.style().polish(row)
            row.update()

    def toggle_layer_visibility(self, layer_id):
        layer = self._find_layer(layer_id)
        if layer is None:
            return
        layer['visible'] = not layer['visible']
        layer['actor'].SetVisibility(layer['visible'])
        layer['eye'].setText('◉' if layer['visible'] else '○')
        layer['eye'].setToolTip('隐藏图层' if layer['visible'] else '显示图层')
        self.vtk_widget.GetRenderWindow().Render()

    def show_and_activate_layer(self, layer_id):
        layer = self._find_layer(layer_id)
        if layer is not None and not layer['visible']:
            self.toggle_layer_visibility(layer_id)
        self.activate_layer(layer_id)

    def show_layer_menu(self, layer_id, button):
        menu = QMenu(self)
        export_action = menu.addAction('导出当前图层 PCD…')
        save_action = menu.addAction('保存原始 PCD…')
        selected = menu.exec_(button.mapToGlobal(QPoint(button.width(), 0)))
        if selected == export_action:
            self.export_layer(layer_id)
        elif selected == save_action:
            self.save_layer_source(layer_id)

    def _find_layer(self, layer_id):
        return next((layer for layer in self.layers if layer['id'] == layer_id), None)

    def export_layer(self, layer_id):
        layer = self._find_layer(layer_id)
        if layer is None:
            return
        suggested = f"{Path(layer['filename']).stem}_display.pcd"
        path, _ = QFileDialog.getSaveFileName(self, '导出当前图层点云', suggested, 'PCD 点云 (*.pcd)')
        if not path:
            return
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(layer['display_points'])
        if not o3d.io.write_point_cloud(path, cloud, write_ascii=False, compressed=False):
            QMessageBox.warning(self, '导出失败', '无法写入指定的 PCD 文件。')

    def save_layer_source(self, layer_id):
        layer = self._find_layer(layer_id)
        if layer is None:
            return
        suggested = Path(layer['filename']).name
        path, _ = QFileDialog.getSaveFileName(self, '保存原始 PCD', suggested, 'PCD 点云 (*.pcd)')
        if not path:
            return
        source_path = layer.get('source_path')
        if source_path and Path(source_path).is_file() and 'edit_points' not in layer:
            try:
                if Path(source_path).resolve() != Path(path).resolve():
                    shutil.copyfile(source_path, path)
            except OSError as error:
                QMessageBox.warning(self, '保存失败', str(error))
            return
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(layer.get('edit_points', layer['source_points']))
        if not o3d.io.write_point_cloud(path, cloud, write_ascii=False, compressed=False):
            QMessageBox.warning(self, '保存失败', '无法写入指定的 PCD 文件。')

    def set_cloud(self, points, title, filename, full_z=None, total_points=None, source_path=None):
        points = np.asarray(points, dtype=np.float32)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            raise ValueError('文件中没有有效的 XYZ 点。')
        source_z = full_z if full_z is not None else points[:, 2]
        z_min, z_max = float(np.nanmin(source_z)), float(np.nanmax(source_z))
        if z_max - z_min < 1e-6:
            z_max = z_min + 0.001
        bounds_min, bounds_max = np.nanmin(points, axis=0), np.nanmax(points, axis=0)
        bounds_min[2], bounds_max[2] = z_min, z_max
        axis_bounds = {axis: (float(bounds_min[i]), float(bounds_max[i])) for i, axis in enumerate(('x', 'y', 'z'))}

        if not self.layers:
            actor, mapper = self.points_actor, self.mapper
        else:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetLookupTable(self.lookup)
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray('Elevation')
            mapper.SetScalarVisibility(True)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetPointSize(self.point_size)
            self.renderer.AddActor(actor)
        layer = {
            'id': self.next_layer_id, 'title': title, 'filename': filename,
            'source_path': source_path, 'source_points': points,
            'display_points': points.copy(), 'full_z': source_z,
            'total_points': int(total_points or len(source_z)),
            'axis_bounds': axis_bounds, 'axis_ranges': dict(axis_bounds),
            'visible': True, 'actor': actor, 'mapper': mapper,
            'row': None,
        }
        self.next_layer_id += 1
        self.layers.append(layer)
        self._add_layer_row(layer)
        self.activate_layer(layer['id'], reset_camera=True)

    def activate_layer(self, layer_id, reset_camera=False):
        layer = self._find_layer(layer_id)
        if layer is None:
            return
        previous = self._find_layer(self.active_layer_id) if self.active_layer_id is not None else None
        if previous is not layer:
            self.point_edit_mode = None
            self.region_pick_points.clear()
            self.region_pick_scope = None
            self.selected_hull_equations = None
            self.selected_region_vertices = None
            self._clear_region_markers()
            self.completion_base_points = None
            self.completion_markers.clear()
            self._set_point_pick_mode(False)
        if previous is not None:
            previous['display_points'] = self.points.copy()
            previous['axis_ranges'] = dict(self.axis_ranges)
            previous['mapper'].RemoveAllClippingPlanes()
        self.filter_request_id += 1
        self.active_layer_id = layer_id
        self.points_actor, self.mapper = layer['actor'], layer['mapper']
        self.render_source_points = layer['source_points']
        self.full_z = layer['full_z']
        self.total_points = layer['total_points']
        self.z_min = float(np.nanmin(self.full_z))
        self.z_max = float(np.nanmax(self.full_z))
        if self.z_max - self.z_min < 1e-6:
            self.z_max = self.z_min + 0.001
        self.axis_bounds = dict(layer['axis_bounds'])
        self.axis_ranges = dict(layer['axis_ranges'])
        self.mapper.SetLookupTable(self.lookup)
        self.mapper.SetScalarRange(self.z_min, self.z_max)
        self.mapper.RemoveAllClippingPlanes()
        self.lookup.SetTableRange(self.z_min, self.z_max)
        for axis, low_plane, high_plane, low_normal, high_normal in (
            ('x', self.x_low_plane, self.x_high_plane, (1, 0, 0), (-1, 0, 0)),
            ('y', self.y_low_plane, self.y_high_plane, (0, 1, 0), (0, -1, 0)),
            ('z', self.low_plane, self.high_plane, (0, 0, 1), (0, 0, -1)),
        ):
            low_plane.SetNormal(*low_normal)
            high_plane.SetNormal(*high_normal)
            low_plane.SetOrigin(*[self.axis_ranges[axis][0] if j == 'xyz'.index(axis) else 0 for j in range(3)])
            high_plane.SetOrigin(*[self.axis_ranges[axis][1] if j == 'xyz'.index(axis) else 0 for j in range(3)])
            self.mapper.AddClippingPlane(low_plane)
            self.mapper.AddClippingPlane(high_plane)
        self.points_actor.SetVisibility(layer['visible'])
        self._set_render_points(layer['display_points'])
        self.header_file.setText(layer['title'])
        self.layer_name.setText(layer['filename'])
        points = self.render_source_points
        self.layer_points.setText(f"{fmt_number(self.total_points)} 点   ·   XYZ")
        if self.total_points > len(points):
            self.layer_points.setText(f"{fmt_number(self.total_points)} 点   ·   视图抽样 {fmt_number(len(points))}")
        self.metric_points.setText(fmt_number(self.total_points))
        self.metric_height.setText(f'{self.z_min:.1f} — {self.z_max:.1f} m')
        self._refresh_layer_selection()
        self._set_slider_extent()
        self._update_visible_stats()
        self.render_count.setText(f'渲染 {fmt_number(len(self.points))} 点 · 原始 {fmt_number(self.total_points)} 点')
        self.filter_status.setText('选中图层可编辑；滤波需点击“重新计算滤波”应用。')
        if reset_camera or previous is not layer:
            self.reset_camera()
        else:
            self.vtk_widget.GetRenderWindow().Render()

    def _set_render_points(self, points):
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        active = self._find_layer(self.active_layer_id) if self.active_layer_id is not None else None
        if active is not None:
            active['display_points'] = self.points.copy()
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(self.points, deep=True))
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_points)
        vertices = vtk.vtkVertexGlyphFilter()
        vertices.SetInputData(poly)
        vertices.Update()
        z_scalars = numpy_to_vtk(np.ascontiguousarray(self.points[:, 2]), deep=True)
        z_scalars.SetName('Elevation')
        vertices.GetOutput().GetPointData().SetScalars(z_scalars)
        self.mapper.SetInputConnection(vertices.GetOutputPort())
        self.mapper.SetScalarRange(self.z_min, self.z_max)
        self.points_actor.GetProperty().SetPointSize(self.point_size)

    def _set_slider_extent(self):
        for axis, slider in (('x', self.x_range_slider), ('y', self.y_range_slider), ('z', self.z_range_slider)):
            slider.set_range(0, 1000)
            low, high = self.axis_bounds[axis]
            suffix = ' m'
            getattr(self, f'{axis}_overlay_low').setText(f'{low:.1f}{suffix}')
            getattr(self, f'{axis}_overlay_high').setText(f'{high:.1f}{suffix}')

    def apply_range(self, low, high):
        self.apply_axis_range('z', low, high)

    def apply_axis_range(self, axis, low, high):
        if self.points is None:
            return
        min_value, max_value = self.axis_bounds[axis]
        span = max_value - min_value
        low_value = min_value + span * low / 1000
        high_value = min_value + span * high / 1000
        self.axis_ranges[axis] = (low_value, high_value)
        planes = {
            'x': (self.x_low_plane, self.x_high_plane, 0),
            'y': (self.y_low_plane, self.y_high_plane, 1),
            'z': (self.low_plane, self.high_plane, 2),
        }
        low_plane, high_plane, dimension = planes[axis]
        origin_low, origin_high = [0.0] * 3, [0.0] * 3
        origin_low[dimension], origin_high[dimension] = low_value, high_value
        low_plane.SetOrigin(*origin_low)
        high_plane.SetOrigin(*origin_high)
        getattr(self, f'{axis}_overlay_low').setText(f'{low_value:.1f} m')
        getattr(self, f'{axis}_overlay_high').setText(f'{high_value:.1f} m')
        self._update_visible_stats()
        self.vtk_widget.GetRenderWindow().Render()

    def _update_visible_stats(self):
        if self.render_source_points is None or not self.total_points:
            return
        visible_sample_count = len(self.current_visible_source_points())
        visible = round(visible_sample_count * self.total_points / max(1, len(self.render_source_points)))
        self.visible_count.setText(fmt_number(visible))
        percent = visible / self.total_points * 100
        self.visible_percent.setText(f'{percent:.1f}% 点云')

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, '导入 PCD 点云', str(Path.home()), '点云文件 (*.pcd);;所有文件 (*)')
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            points, full_z, total = load_pcd_data(path)
            self.set_cloud(points, Path(path).stem, Path(path).name, full_z, total, path)
        except Exception as error:
            QMessageBox.critical(self, '导入失败', f'无法载入该 PCD 文件：\n{error}')
        finally:
            QApplication.restoreOverrideCursor()

    def reset_range(self):
        for axis, slider in (('x', self.x_range_slider), ('y', self.y_range_slider), ('z', self.z_range_slider)):
            slider.set_range(0, 1000)
            self.apply_axis_range(axis, 0, 1000)

    def reset_camera(self):
        camera = self.renderer.GetActiveCamera()
        bounds = self.points_actor.GetBounds()
        center = [(bounds[i] + bounds[i + 1]) * 0.5 for i in (0, 2, 4)]
        radius = max(1.0, np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]) * 0.5)
        distance = radius * 2.7
        camera.SetViewUp(0, 0, 1)
        camera.SetFocalPoint(*center)
        camera.SetPosition(center[0] + distance * 0.75, center[1] - distance * 0.85, center[2] + distance * 0.95)
        camera.SetViewAngle(34)
        if self.orthographic:
            camera.ParallelProjectionOn()
            camera.SetParallelScale(radius * 0.95)
        else:
            camera.ParallelProjectionOff()
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def set_top_view(self):
        bounds = self.points_actor.GetBounds()
        center = [(bounds[i] + bounds[i + 1]) * 0.5 for i in (0, 2, 4)]
        radius = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 0.5
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(center[0], center[1], center[2] + max(radius * 2.5, 10))
        camera.SetViewUp(0, 1, 0)
        camera.ParallelProjectionOn()
        camera.SetParallelScale(max(radius * 1.08, 1))
        self.orthographic = True
        self.view_label.setText('俯视  ·  Z 轴高度着色')
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def start_held_rotation(self, direction):
        self.rotate_direction = direction
        self.rotate_clock.start()
        self.rotate_timer.start()

    def stop_held_rotation(self):
        self.rotate_timer.stop()
        self.rotate_direction = 0

    def _rotate_held_view(self):
        if not self.rotate_direction or not self.rotate_clock.isValid():
            return
        seconds = self.rotate_clock.restart() / 1000
        self.rotate_around_z(self.rotate_direction * 10.0 * seconds)

    def rotate_around_z(self, degrees):
        camera = self.renderer.GetActiveCamera()
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        position = np.asarray(camera.GetPosition(), dtype=float)
        view_up = np.asarray(camera.GetViewUp(), dtype=float)
        angle = np.deg2rad(degrees)
        rotation = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
        offset = position - focal
        offset[:2] = rotation @ offset[:2]
        view_up[:2] = rotation @ view_up[:2]
        camera.SetPosition(*(focal + offset))
        camera.SetViewUp(*view_up)
        camera.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def toggle_grid(self):
        self.grid_visible = not self.grid_visible
        self.grid_actor.SetVisibility(self.grid_visible)
        self.vtk_widget.GetRenderWindow().Render()

    def toggle_projection(self):
        self.orthographic = not self.orthographic
        camera = self.renderer.GetActiveCamera()
        if self.orthographic:
            camera.ParallelProjectionOn()
            camera.SetParallelScale(52)
            self.view_label.setText('正交  ·  Z 轴高度着色')
        else:
            camera.ParallelProjectionOff()
            self.view_label.setText('俯视  ·  Z 轴高度着色')
        self.vtk_widget.GetRenderWindow().Render()

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def set_point_size(self, size):
        self.point_size = float(size)
        self.points_actor.GetProperty().SetPointSize(self.point_size)
        self.size_text.setText(f'{size} px')
        self.vtk_widget.GetRenderWindow().Render()

    def export_image(self):
        path, _ = QFileDialog.getSaveFileName(self, '导出视图截图', 'pointcloud-view.png', 'PNG 图片 (*.png)')
        if not path:
            return
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(self.vtk_widget.GetRenderWindow())
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('taffy-editor')
    app.setApplicationDisplayName(f'taffy编辑器 {APP_VERSION}')
    app.setDesktopFileName('taffy-editor')
    app.setWindowIcon(QIcon(str(Path(__file__).parent / 'assets' / 'tafei-pcd-editor.png')))
    app.setFont(QFont('Noto Sans CJK SC', 10))
    app.setStyleSheet(APP_STYLE)
    window = PointCloudWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
