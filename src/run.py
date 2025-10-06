# Author : Rezky M. Kam
# Do not use for real medical application. Only for reference.

import sys
from PyQt6.QtWidgets import QApplication
from ui import SkinSpectorApp


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SkinSpectorApp()
    window.show()
    sys.exit(app.exec())