"""CLI entrypoint for the ultrasound imaging toolkit demos."""

from ultrasound.demo import _parse_args, main

if __name__ == "__main__":
    args = _parse_args()
    main(demos=args.demo, output_dir=args.output_dir, data_dir=args.data_dir)
