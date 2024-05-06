import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget
from transformers import BartForConditionalGeneration, BartTokenizer

class TextSummarizerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Text Summarizer")
        self.setGeometry(100, 100, 600, 400)

        self.text_input = QTextEdit(self)
        self.text_input.setPlaceholderText("Enter text to summarize...")
        self.text_input.setGeometry(20, 20, 560, 200)

        self.summarize_button = QPushButton("Summarize", self)
        self.summarize_button.setGeometry(250, 240, 100, 40)
        self.summarize_button.clicked.connect(self.summarize_text)

        self.summarized_output = QTextEdit(self)
        self.summarized_output.setReadOnly(True)
        self.summarized_output.setGeometry(20, 290, 560, 100)

    def summarize_text(self):
        input_text = self.text_input.toPlainText()

        
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        
        inputs = tokenizer.batch_encode_plus([input_text], return_tensors='pt', max_length=1024, truncation=True)

        
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=60, max_length=200, early_stopping=True)

        # Decode the summary
        summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        self.summarized_output.setPlainText(summarized_text)

def run_app():
    app = QApplication(sys.argv)
    main_window = TextSummarizerApp()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_app()
