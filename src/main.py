"""
「衣装を、別のアバターへ、自動で、きれいに。」
このシステムは、Blenderを使って衣装モデルを新しい素体(ベースアバター)に合わせて自動的に調整(リターゲット)するツールです。
---
1. 入力: 「着せたい衣装」と「着せたい素体」を用意します。
2. 処理: システムが衣装の形を解析し、素体の体型に合わせて変形させます。
3. 出力: Unityですぐに使える、スキニング済みのFBXファイルを生成します。
---
ただ拡大縮小するだけではありません。「衣装の素材感」を大切にします。
衣装をパーツごとに分解し、それが「柔らかい服」なのか「硬いアクセサリー」なのかを自動で判断します。
- 硬いパーツ(バックル、宝石など):
  形を歪ませず、位置とサイズだけを合わせます。「飾りが伸びてへしゃげる」のを防ぎます。
  (技術: OBB判定 + 類似変換)
- 柔らかいパーツ(布地、袖など):
  素体の表面に合わせて、滑らかに変形させます。シワや形状の流れを壊さずにフィットさせます。
  (技術: RBF補間)
衣装が変形したら、素体のボーンウェイト(動く仕組み)を衣装にコピーします。
これにより、VRChatなどでアバターが動いたとき、衣装も自然に追従します。
---
- 物理シミュレーション: 布が揺れたり落ちたりする計算はしません。あくまで「モデリングとしての変形」を行います。
- 体型の変更: 素体(ベースアバター)の形は絶対に変えません。衣装側を合わせます。
---
「数学の力を使って、手作業のような丁寧さで、衣装を自動フィッティングするシステム」 です。
"""

import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from application.retargeter import OutfitRetargeter


def parse_args():
    parser = argparse.ArgumentParser(description="Outfit Retargeting System")
    parser.add_argument("--input", required=False, help="Input clothing FBX file path (optional if in config)")
    parser.add_argument("--output", required=False, help="Output FBX file path")
    parser.add_argument("--base", required=True, help="Base Blender file path")
    parser.add_argument("--base-fbx", required=True, help="Semicolon-separated list of base avatar FBX file paths")
    parser.add_argument("--config", required=True, help="Semicolon-separated list of config file paths")
    parser.add_argument("--hips-position", type=str, help="Target Hips bone world position (x,y,z format)")
    parser.add_argument("--blend-shapes", type=str, help="Semicolon-separated list of blend shape labels to apply")
    parser.add_argument("--cloth-metadata", type=str, help="Path to cloth metadata JSON file")
    parser.add_argument("--mesh-material-data", type=str, help="Path to mesh material data JSON file")
    parser.add_argument("--init-pose", required=True, help="Initial pose data JSON file path")
    parser.add_argument("--target-meshes", required=False, help="Semicolon-separated list of mesh names to process")
    parser.add_argument(
        "--no-subdivision", action="store_true", help="Disable subdivision during DeformationField deformation"
    )
    parser.add_argument("--no-triangle", action="store_true", help="Disable mesh triangulation")
    parser.add_argument(
        "--blend-shape-values", type=str, help="Semicolon-separated list of float values for blend shape intensities"
    )
    parser.add_argument(
        "--blend-shape-mappings", type=str, help="Semicolon-separated mappings of label,customName pairs"
    )
    parser.add_argument("--name-conv", type=str, help="Path to bone name conversion JSON file")
    parser.add_argument("--mesh-renderers", type=str, help="Semicolon-separated list of meshObject,parentObject pairs")
    parser.add_argument("--shape-name-file", type=str, help="Path to JSON file containing BlendShape names per mesh")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    args = parse_args()
    retargeter = OutfitRetargeter()
    retargeter.execute(args)


if __name__ == "__main__":
    main()
