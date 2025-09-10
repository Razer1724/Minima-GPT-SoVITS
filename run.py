from ui.ui import create_ui
from config import is_share, webui_port_main

if __name__ == "__main__":
    app = create_ui()
    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=webui_port_main,
    )