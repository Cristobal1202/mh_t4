if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["boa", "tlbo", "hs", "de"])
    parser.add_argument("--mode", type=str, choices=["baseline", "tuning"])
    args = parser.parse_args()

    if args.mode == "baseline":
        from experiments.run_experiments import run_baseline
        run_baseline(args.algo)
    elif args.mode == "tuning":
        from tuning.tuning_boa import run_tuning  # ejemplo
        run_tuning()
