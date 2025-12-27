import sys
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QScrollArea,
    QMessageBox, QGroupBox, QProgressBar, QTabWidget, QCheckBox,
    QSizePolicy  
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, pyqtSignal, QThread

from backends import SkinSpector
from utils.worker import LLMWorker, DocumentProcessingWorker, StreamEmitter


class SkinSpectorApp(QWidget):
    append_text_signal = pyqtSignal(str)
    update_progress_signal = pyqtSignal(int, str)
    update_kb_stats_signal = pyqtSignal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SKINspector")
        self.setGeometry(200, 200, 1200, 500)

        # Backend
        self.backend = SkinSpector()
        
        # Threading
        self.llm_thread = None
        self.llm_worker = None
        self.doc_thread = None
        self.doc_worker = None
        
        # UI setup
        self.setup_ui()
        
        # Connect signals
        self.append_text_signal.connect(self.append_text)
        self.update_progress_signal.connect(self.update_progress)
        self.update_kb_stats_signal.connect(self.update_kb_stats)
        
        # Load initial KB stats
        self.refresh_kb_stats()

    def setup_ui(self) -> None:
        """Initialize and arrange UI components with tabs"""
        main_layout = QVBoxLayout()
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Analysis Tab
        analysis_tab = self.create_analysis_tab()
        # Knowledge Base Tab
        kb_tab = self.create_knowledge_base_tab()
        # Terminal Logging Tab
        logs_tab = self.create_logs_tab()
        
        tab_widget.addTab(analysis_tab, "Skin Analysis")
        tab_widget.addTab(kb_tab, "Knowledge Base")
        tab_widget.addTab(logs_tab, "Logs")
        
        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)

    def create_analysis_tab(self) -> QWidget:
        """Create the main analysis tab"""
        tab = QWidget()
        main_layout = QHBoxLayout()
        
        # Left panel - Image and controls
        left_panel = QVBoxLayout()
        
        self.image_label = QLabel("No image uploaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedWidth(400)

        self.resolution_label = QLabel("Resolution: N/A")
        self.resolution_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # RAG toggle
        rag_group = QGroupBox("Analysis Options")
        rag_layout = QVBoxLayout()
        
        self.rag_checkbox = QCheckBox("Use Medical Knowledge Base (RAG)")
        self.rag_checkbox.setChecked(True)
        self.rag_checkbox.setToolTip("Enhance analysis with medical literature from your knowledge base")
        rag_layout.addWidget(self.rag_checkbox)
        
        self.kb_stats_label = QLabel("Knowledge Base: Loading...")
        rag_layout.addWidget(self.kb_stats_label)
        
        rag_group.setLayout(rag_layout)

        # Upload buttons group
        upload_group = QGroupBox("Upload")
        upload_layout = QVBoxLayout()
        
        self.upload_btn = QPushButton("Upload Skin Image")
        self.upload_btn.clicked.connect(self.upload_image)
        
        upload_layout.addWidget(self.upload_btn)
        upload_group.setLayout(upload_layout)
        
        # Stop button
        self.stop_btn = QPushButton("Stop Analysis")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(100)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignLeft) 
        self.progress_bar.setVisible(False)
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        
        left_panel.addWidget(self.image_label)
        left_panel.addWidget(self.resolution_label)
        left_panel.addWidget(rag_group)
        left_panel.addWidget(upload_group)
        left_panel.addWidget(self.stop_btn)
        left_panel.addStretch()
        left_panel.addWidget(self.progress_label)
        left_panel.addWidget(self.progress_bar)
        left_panel.addStretch()

        # Right panel - Text output
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Analysis Output:"))
        
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.text_output)
        
        right_panel.addWidget(scroll_area)

        # Layout assembly
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)
        tab.setLayout(main_layout)
        
        return tab

    def create_knowledge_base_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Knowledge Base Controls
        controls_group = QGroupBox("Knowledge Base Management")
        controls_layout = QVBoxLayout()
        
        # Stats
        stats_layout = QHBoxLayout()
        self.kb_stats_full_label = QLabel("Total documents: Loading...")
        stats_layout.addWidget(self.kb_stats_full_label)
        stats_layout.addStretch()
        
        # Document upload
        upload_layout = QHBoxLayout()
        self.upload_doc_btn = QPushButton("Upload Medical Document")
        self.upload_doc_btn.clicked.connect(self.upload_document)
        upload_layout.addWidget(self.upload_doc_btn)
        
        self.clear_kb_btn = QPushButton("Clear Knowledge Base")
        self.clear_kb_btn.clicked.connect(self.clear_knowledge_base)
        upload_layout.addWidget(self.clear_kb_btn)
        
        self.refresh_kb_btn = QPushButton("Refresh Stats")
        self.refresh_kb_btn.clicked.connect(self.refresh_kb_stats)
        upload_layout.addWidget(self.refresh_kb_btn)
        
        # Document processing progress
        self.doc_progress_bar = QProgressBar()
        self.doc_progress_bar.setVisible(False)
        self.doc_progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.doc_progress_bar.setMinimumHeight(25)
        
        self.doc_progress_label = QLabel("")
        self.doc_progress_label.setVisible(False)
        
        controls_layout.addLayout(stats_layout)
        controls_layout.addLayout(upload_layout)
        controls_layout.addWidget(self.doc_progress_bar)
        controls_layout.addWidget(self.doc_progress_label)
        controls_group.setLayout(controls_layout)
        
        # Supported formats info
        info_label = QLabel(
            "Supported formats: PDF, DOCX, DOC, TXT\n"
            "Upload medical textbooks, research papers, or clinical guidelines to enhance analysis."
        )
        info_label.setStyleSheet("color: #111; font-size: 11px;")
        
        layout.addWidget(controls_group)
        layout.addWidget(info_label)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def create_logs_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()

        self.logs_output = QTextEdit()
        self.logs_output.setReadOnly(True)

        layout.addWidget(self.logs_output)
        tab.setLayout(layout)

        # Redirect stdout and stderr to this widget
        self.stdout_emitter = StreamEmitter()
        self.stderr_emitter = StreamEmitter()
        self.stdout_emitter.new_text.connect(self.append_log_text)
        self.stderr_emitter.new_text.connect(self.append_log_text)
        
        sys.stdout = self.stdout_emitter
        sys.stderr = self.stderr_emitter

        return tab
    
    def append_log_text(self, text: str):
        cursor = self.logs_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.logs_output.setTextCursor(cursor)
        self.logs_output.ensureCursorVisible()

    def refresh_kb_stats(self):
        stats = self.backend.get_kb_stats()
        self.update_kb_stats_signal.emit(stats)

    def update_kb_stats(self, stats: dict):
        total_docs = stats.get("total_documents", 0)
        stats_text = f"Knowledge Base: {total_docs} documents"
        full_stats_text = f"Total documents: {total_docs} | Embedding model: {stats.get('embedding_model', 'N/A')}"
        
        self.kb_stats_label.setText(stats_text)
        self.kb_stats_full_label.setText(full_stats_text)

    def upload_document(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Upload Medical Document", "", 
            "Documents (*.pdf *.docx *.doc *.txt);;PDF Files (*.pdf);;Word Documents (*.docx *.doc);;Text Files (*.txt)"
        )

        if not file_path:
            return

        if not self.backend.validate_document_file(file_path):
            QMessageBox.warning(self, "Invalid File", "Please select a valid document file.")
            return

        # Start document processing in background thread
        self.start_document_processing(file_path)

    def start_document_processing(self, file_path: str):
        # Stop any existing processing
        self.stop_document_processing()
        
        # Setup thread and worker
        self.doc_thread = QThread()
        self.doc_worker = DocumentProcessingWorker(self.backend.get_rag_system(), file_path)
        self.doc_worker.moveToThread(self.doc_thread)
        
        # Connect signals
        self.doc_thread.started.connect(self.doc_worker.process_document)
        self.doc_worker.finished.connect(self.doc_thread.quit)
        self.doc_worker.finished.connect(self.doc_worker.deleteLater)
        self.doc_thread.finished.connect(self.doc_thread.deleteLater)
        self.doc_worker.progress_update.connect(self.update_document_progress)
        self.doc_worker.error.connect(self.handle_document_error)
        
        # Update UI
        self.upload_doc_btn.setEnabled(False)
        self.clear_kb_btn.setEnabled(False)
        self.doc_progress_bar.setVisible(True)
        self.doc_progress_label.setVisible(True)
        
        self.doc_thread.finished.connect(self.on_document_processing_finished)
        
        # Start thread
        self.doc_thread.start()

    def stop_document_processing(self):
        if self.doc_worker:
            self.doc_worker.stop()
        if self.doc_thread and self.doc_thread.isRunning():
            self.doc_thread.quit()
            self.doc_thread.wait(1000)

    def on_document_processing_finished(self):
        self.upload_doc_btn.setEnabled(True)
        self.clear_kb_btn.setEnabled(True)
        self.doc_progress_bar.setVisible(False)
        self.doc_progress_label.setVisible(False)
        self.doc_worker = None
        self.doc_thread = None
        
        # Refresh stats
        self.refresh_kb_stats()

    def update_document_progress(self, value: int, text: str):
        self.doc_progress_bar.setValue(value)
        self.doc_progress_label.setText(text)

    def handle_document_error(self, error_msg: str):
        QMessageBox.critical(self, "Document Processing Error", f"Failed to process document: {error_msg}")
        self.on_document_processing_finished()

    def clear_knowledge_base(self):
        reply = QMessageBox.question(
            self, "Clear Knowledge Base", 
            "Are you sure you want to clear the entire knowledge base? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.backend.clear_knowledge_base()
                self.refresh_kb_stats()
                QMessageBox.information(self, "Success", "Knowledge base cleared successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear knowledge base: {e}")

    def upload_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return

        if not self.backend.validate_image_file(file_path):
            QMessageBox.warning(self, "Invalid File", "Please select a valid image file.")
            return

        self.backend.set_image_path(file_path)

        # Show scaled image
        pixmap = QPixmap(file_path).scaled(
            400, 400, Qt.AspectRatioMode.KeepAspectRatio
        )
        self.image_label.setPixmap(pixmap)

        # Show resolution (original, not scaled)
        original_pixmap = QPixmap(file_path)
        width = original_pixmap.width()
        height = original_pixmap.height()
        self.resolution_label.setText(f"Resolution: {width} x {height} ({width*height:,} pixels)")

        # Clear old text output
        self.text_output.clear()

        # Start background thread for LLM processing
        self.start_image_processing()

    def start_image_processing(self) -> None:
        # Stop any existing analysis
        self.stop_analysis()
        
        # Setup thread and worker
        self.llm_thread = QThread()
        self.llm_worker = LLMWorker(
            llm_vl      =  self.backend.get_llm_vl(),
            llm         =  self.backend.get_llm(), 
            file_path   =  self.backend.get_image_path(), 
            rag_system  =  self.backend.get_rag_system() if self.rag_checkbox.isChecked() else None,
            use_rag     =  self.rag_checkbox.isChecked()
        )
        
        # Move worker to thread
        self.llm_worker.moveToThread(self.llm_thread)
        
        # Connect signals
        self.llm_thread.started.connect(self.llm_worker.process_image)
        self.llm_worker.finished.connect(self.llm_thread.quit)
        self.llm_worker.finished.connect(self.llm_worker.deleteLater)
        self.llm_thread.finished.connect(self.llm_thread.deleteLater)
        self.llm_worker.text_chunk.connect(self.append_text_signal.emit)
        self.llm_worker.progress_update.connect(self.update_progress_signal)
        self.llm_worker.error.connect(self.handle_error)
        
        # Update UI state
        self.upload_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.llm_thread.finished.connect(self.on_analysis_finished)
        
        # Start thread
        self.llm_thread.start()

    def stop_analysis(self):
        if self.llm_worker:
            self.llm_worker.stop()
        if self.llm_thread and self.llm_thread.isRunning():
            self.llm_thread.quit()
            self.llm_thread.wait(1000)

    def on_analysis_finished(self):
        self.upload_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.llm_worker = None
        self.llm_thread = None

    def handle_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"Analysis failed: {error_msg}")
        self.on_analysis_finished()

    def append_text(self, text: str) -> None:
        cursor = self.text_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.text_output.setTextCursor(cursor)
        self.text_output.ensureCursorVisible()

    def update_progress(self, value: int, text: str):
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(value)
        self.progress_label.setText(text)
        
        if value >= 100:
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)

    def closeEvent(self, event):
        self.stop_analysis()
        self.stop_document_processing()
        event.accept()