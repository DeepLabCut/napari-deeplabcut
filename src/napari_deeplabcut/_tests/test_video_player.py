import sys
from napari_deeplabcut._widgets import VideoPlayer
from qtpy.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer(parent=None, video_path='/Users/jessy/Desktop/E2.mov')
    player.media_player.pause()
    player.show()
    app.exec()
    app.quit()
