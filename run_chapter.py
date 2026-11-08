"""Run a chapter or an added activity using the current Python environment.

Examples: python run_chapter.py 1
          python run_chapter.py 6 --activity
          python run_chapter.py --all
"""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('chapter', type=int, nargs='?', choices=range(1,17))
    parser.add_argument('--all', action='store_true', help='run all sixteen chapter scripts')
    parser.add_argument('--activity', action='store_true', help='run the added activity (chapters 1–11)')
    args = parser.parse_args()
    if args.all and args.chapter:
        parser.error('Choose a chapter or --all.')
    if not args.all and args.chapter is None:
        parser.error('Choose a chapter or --all.')
    if args.all and args.activity:
        script = ROOT / 'python' / 'activities' / 'experiments.py'
        print('Running all eleven added investigations', flush=True)
        subprocess.run([sys.executable, str(script), '--all'], cwd=script.parent, check=True)
        return
    chapters = range(1,12 if args.activity else 17) if args.all else [args.chapter]
    for chapter in chapters:
        if args.activity:
            if chapter > 11:
                parser.error('Added activities are available for chapters 1–11.')
            script = ROOT / 'python' / 'activities' / 'experiments.py'
            cmd = [sys.executable,str(script),'--chapter',str(chapter)]
        else:
            matches = sorted((ROOT/'python').glob(f'ch{chapter:02d}_*.py'))
            if len(matches) != 1:
                raise SystemExit(f'Expected one script for chapter {chapter}, found {len(matches)}.')
            script=matches[0];cmd=[sys.executable,str(script)]
        print(f'Running chapter {chapter}'+(' activity' if args.activity else ''),flush=True)
        subprocess.run(cmd,cwd=script.parent,check=True)

if __name__ == '__main__':
    main()
