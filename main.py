from pathlib import Path
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
import soundfile as sf

BERT_MODEL_NAME = "ku-nlp/deberta-v2-large-japanese-char-wwm"
MODEL_ASSETS_DIR = Path("model_assets")
MODEL_FILE = "downer/downer_e100_s7000.safetensors"
CONFIG_FILE = "downer/config.json"
STYLE_FILE = "downer/style_vectors.npy"


def load_japanese_bert() -> None:
    # Load Japanese BERT model and tokenizer (auto-downloaded from Hugging Face on first run)
    bert_models.load_model(Languages.JP, BERT_MODEL_NAME).float()
    bert_models.load_tokenizer(Languages.JP, BERT_MODEL_NAME)


def build_tts_model(assets_root: Path, device: str) -> TTSModel:
    # Build a TTSModel from the model assets
    return TTSModel(
        model_path=assets_root / MODEL_FILE,
        config_path=assets_root / CONFIG_FILE,
        style_vec_path=assets_root / STYLE_FILE,
        device=device,
    )


def synthesize(model: TTSModel, text: str, output_path: Path) -> None:
    # Run inference and write the resulting audio to a wav file
    sr, audio = model.infer(text=text, style="neutral", length=0.9)
    sf.write(output_path, audio, sr)


if __name__ == "__main__":
    load_japanese_bert()
    model = build_tts_model(MODEL_ASSETS_DIR, device="cuda")  # use "cpu" if no GPU available
    output_path = Path("output.wav")
    text = (
        "昔々、あるところにおじいさんとおばあさんが住んでいました。"
        "ある日、おばあさんが川で洗濯をしていると、大きな桃がどんぶらこどんぶらこと流れてきました。"
        "家に持ち帰り割ってみると、なかから元気な男の子が飛び出してきました。"
        "二人は大喜びで、その子を「桃太郎」と名付け、大切に育てました。\n\n"
        "桃太郎はすくすくと成長し、心優しく力の強い若者になりました。"
        "ある日、村では鬼ヶ島に住む鬼たちが人々を苦しめているという噂を耳にします。"
        "桃太郎は「僕が鬼退治に行きます」と決意し、おばあさんが作ってくれたきびだんごを腰に下げて旅立ちました。\n\n"
        "道中、桃太郎は犬に出会います。「きびだんごを一つください、おともします」と犬は言い、仲間になりました。"
        "同じように猿、そして雉も、きびだんごと引き換えに家来となり、四人は力を合わせて鬼ヶ島を目指しました。\n\n"
        "鬼ヶ島に着くと、犬が鬼の門に噛みつき、猿が高い塀をみがるに乗り越え、雉が空から鬼の頭をつつきました。"
        "桃太郎も刀を手に鬼の大将と堂々と渡り合い、ついに鬼たちを降参させました。"
        "鬼は今までに奪った財宝を差し出し、二度と悪さをしないと誓いました。\n\n"
        "桃太郎は仲間たちと共にたからものを村へ持ち帰り、おじいさんとおばあさんに献上しました。"
        "村人たちは平和を取り戻し、桃太郎とその仲間たちの勇気をいつまでも語り継いだということです。"
    )
    synthesize(model, text=text, output_path=output_path)
    print(f"{output_path} に保存しました")