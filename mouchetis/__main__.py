"""mouchetis photo.jpg | mouchetis bubbles ... | mouchetis audit ..."""
import os
import sys

# MediaPipe / TFLite C++ logs drown the useful output
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

USAGE = """usage:
  mouchetis PHOTO [options]      dot screen with the face erased (mouchetis -h)
  mouchetis bubbles PHOTO ...    speech bubbles on a photo
  mouchetis audit FOLDER ...     check renders before posting them
"""


def main() -> None:
    args = sys.argv[1:]
    if not args:
        sys.exit(USAGE)
    if args[0] in ("bubbles", "audit"):
        cmd = args.pop(0)
        sys.argv = [f"mouchetis {cmd}", *args]
        if cmd == "bubbles":
            from .bubbles import main as run
        else:
            from .audit import main as run
    else:
        sys.argv = ["mouchetis", *args]
        from .stipple import main as run
    run()


if __name__ == "__main__":
    main()
