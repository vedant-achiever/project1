import os
import threading
from django.apps import AppConfig


class AiEngineConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ai_engine'

    def ready(self):
        # Pre-warm Qwen2.5 chat model on server startup
        if os.environ.get('RUN_MAIN') == 'true':
            def _preload():
                try:
                    from .model_loader import get_chat_model
                    get_chat_model()
                except Exception:
                    pass

            threading.Thread(target=_preload, daemon=True).start()
