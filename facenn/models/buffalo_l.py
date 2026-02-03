from facenn.models.arcface import ArcFace

class Buffalo_L(ArcFace):
    """
    Buffalo_L is essentially ArcFace with R50 backbone (IR-SE50).
    It usually comes from the InsightFace Buffalo_L model pack.
    We reuse the ArcFace class but would point to specific 'buffalo_l' weights if available.
    For now, it aliases ArcFace (R50).
    """
    def __init__(self):
        super().__init__()
        self.model_name = "Buffalo_L"
