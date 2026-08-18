"""
衣装リターゲットというのは、地味で根気のいる作業です。
1着につき数分だとしても、持っている衣装が50着あれば、休日の午後が全部潰れてしまいます。
システムは、いわば「全自動ロボット工場」です。
工場のベルトコンベアに、変換したい衣装フォルダを放り込んでおけば、あとは放っておくだけ。
1.  フォルダ内のFBXファイルを全部見つける
2.  ひとつずつ取り出して、リターゲットする
3.  終わったら次のを取り出す
もし途中で「この服はデータが壊れていて読み込めない」というトラブルがあっても、工場は止まりません。
「これはエラーでした」というメモと現場写真(以前説明したエラーファイル)だけ残して、淡々と次の服の処理に進みます。
この機能のおかげで、もはやパソコンの前で進捗バーを見守る必要はありません。
夜寝る前にセットして実行ボタンを押し、朝起きたら変換済みの50着が出来上がっている。
空いた時間は、VRChatでフレンドと遊んだり、新しい改変を考えることに使ってください。
単調な作業は、すべて機械に任せてしまいましょう。
"""
import glob
import os
import subprocess
from datetime import datetime
from domain.models import BatchConfig
SCRIPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
def main():
    config = BatchConfig()
    fbx_files = glob.glob(os.path.join(config.input_dir, "**/*.fbx"), recursive=True)
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for fbx in fbx_files:
        base_name = os.path.splitext(os.path.basename(fbx))[0]
        output_fbx = os.path.join(config.output_dir, f"retargeted_{timestamp}_{base_name}.fbx")
        cmd = [
            config.blender_exe,
            "--background",
            "--python",
            SCRIPT_PATH,
            "--",
            "--base",
            config.base_fbx,
            "--input",
            fbx,
            "--output",
            output_fbx,
            "--config",
            config.config_path,
            "--init-pose",
            config.init_pose,
        ]
        subprocess.run(cmd)
if __name__ == "__main__":
    main()
