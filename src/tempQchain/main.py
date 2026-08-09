import typer

app = typer.Typer(help="TempQChain CLI")


@app.command()
def create_tb_dense(
    save_rules: bool = typer.Option(False, help="Save transitivity rules to file"),
    augment_train: bool = typer.Option(False, help="Augment training set with q-chains"),
):
    """Process TB-Dense data and create training/dev/test splits."""
    import tempQchain.data.create_tb_dense as create_tb_dense

    typer.echo("Processing TB-Dense data...")

    try:
        create_tb_dense.process_tb_dense(
            trans_rules=create_tb_dense.trans_rules, save_rules_to_file=save_rules, augment_train=augment_train
        )
        typer.echo("✅ Data processing completed successfully!")
    except Exception as e:
        typer.echo(f"❌ Error during data processing: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def temporal_fr(
    # Training parameters
    seed: int = typer.Option(42, help="Seed value used for experiment"),
    model: str = typer.Option("bert", help="Model used"),
    epoch: int = typer.Option(10, help="Number of training epochs"),
    lr: float = typer.Option(1e-5, help="Learning rate"),
    weight_decay: float = typer.Option(1e-3, help="Weight decay for AdamW"),
    batch_size: int = typer.Option(16, help="Batch size for training"),
    patience: int = typer.Option(3, help="Patience for early stopping"),
    c_lr: float = typer.Option(0.05, help="Constraint learning rate"),
    c_warmup_iters: int = typer.Option(543, help="Warm up iterations for constraint optimization"),
    c_freq_increase: int = typer.Option(10, help="Update frequency of constrained lagrange multipliers"),
    c_freq_increase_freq: int = typer.Option(1, help="Increase frequency of c_freq_increase"),
    c_lr_decay: int = typer.Option(4, help="Index for constraint learning rate decay strategy"),
    c_lr_decay_param: float = typer.Option(1.0, help="Decay parameter for constraint learning rate decay strategy"),
    # Data parameters
    data_path: str = typer.Option("data/", help="Path to the data folder"),
    # Model parameters
    dropout: bool = typer.Option(False, help="Enable dropout"),
    constraints: bool = typer.Option(False, help="Enable constraints"),
    transitive_enabled: bool = typer.Option(True, help="Enable transitive constraints"),
    inverse_enabled: bool = typer.Option(True, help="Enable inverse constraints"),
    use_class_weights: bool = typer.Option(False, help="Enable class weights for training"),
    # Training method parameters
    pmd: bool = typer.Option(False, help="Use Primal Dual method"),
    beta: float = typer.Option(1.0, help="Beta parameter for PMD"),
    sampling: bool = typer.Option(False, help="Use sampling loss"),
    sampling_size: int = typer.Option(1, help="Sampling size"),
    # Additional options
    cuda: int = typer.Option(0, help="CUDA device number (-1 for CPU)"),
    # Model loading/saving, experiment tracking
    run_name: str = typer.Option(None, help="Run name used for MLflow and saved model"),
    best_model_dir: str = typer.Option("models/", help="Directory name to save model"),
    use_mlflow: bool = typer.Option(False, help="Use MLflow for experiment tracking"),
):
    import argparse

    import tempQchain.temporal_fr as temporal_fr

    args = argparse.Namespace(
        seed=seed,
        model=model,
        epoch=epoch,
        lr=lr,
        weight_decay=weight_decay,
        cuda=cuda,
        batch_size=batch_size,
        data_path=data_path,
        dropout=dropout,
        pmd=pmd,
        beta=beta,
        sampling=sampling,
        sampling_size=sampling_size,
        constraints=constraints,
        transitive_enabled=transitive_enabled,
        inverse_enabled=inverse_enabled,
        best_model_dir=best_model_dir,
        use_mlflow=use_mlflow,
        use_class_weights=use_class_weights,
        patience=patience,
        c_lr=c_lr,
        c_warmup_iters=c_warmup_iters,
        c_freq_increase=c_freq_increase,
        c_freq_increase_freq=c_freq_increase_freq,
        c_lr_decay=c_lr_decay,
        c_lr_decay_param=c_lr_decay_param,
        run_name=run_name,
    )
    temporal_fr.main(args)


@app.command()
def temporal_yn(
    # Training parameters
    seed: int = typer.Option(42, help="Seed value used for experiment"),
    model: str = typer.Option("bert", help="Model used"),
    epoch: int = typer.Option(1, help="Number of training epochs"),
    lr: float = typer.Option(1e-5, help="Learning rate"),
    weight_decay: float = typer.Option(1e-3, help="Weight decay for AdamW"),
    batch_size: int = typer.Option(4, help="Batch size for training"),
    patience: int = typer.Option(3, help="Patience for early stopping"),
    c_lr: float = typer.Option(0.05, help="Constraint learning rate"),
    c_warmup_iters: int = typer.Option(150, help="Warm up iterations for constraint optimization"),
    c_freq_increase: int = typer.Option(5, help="Update frequency of constrained lagrange multipliers"),
    c_freq_increase_freq: int = typer.Option(1, help="Increase frequency of c_freq_increase"),
    c_lr_decay: int = typer.Option(0, help="Index for constraint learning rate decay strategy"),
    c_lr_decay_param: float = typer.Option(1.0, help="Decay parameter for constraint learning rate decay strategy"),
    # Data parameters
    data_path: str = typer.Option("data/", help="Path to the data folder"),
    # Model parameters
    dropout: bool = typer.Option(False, help="Enable dropout"),
    constraints: bool = typer.Option(False, help="Enable constraints"),
    use_class_weights: bool = typer.Option(False, help="Enable class weights for training"),
    # Training method parameters
    pmd: bool = typer.Option(False, help="Use Primal Dual method"),
    beta: float = typer.Option(0.5, help="Beta parameter for PMD"),
    sampling: bool = typer.Option(False, help="Use sampling loss"),
    sampling_size: int = typer.Option(1, help="Sampling size"),
    # Additional options
    cuda: int = typer.Option(0, help="CUDA device number (-1 for CPU)"),
    # Model loading/saving, experiment tracking
    run_name: str = typer.Option(None, help="Run name used for MLflow and saved model"),
    best_model_dir: str = typer.Option("models/", help="Directory name to save model"),
    use_mlflow: bool = typer.Option(False, help="Use MLflow for experiment tracking"),
):
    import argparse

    import tempQchain.temporal_yn as temporal_yn

    args = argparse.Namespace(
        seed=seed,
        model=model,
        epoch=epoch,
        lr=lr,
        weight_decay=weight_decay,
        cuda=cuda,
        batch_size=batch_size,
        data_path=data_path,
        dropout=dropout,
        pmd=pmd,
        beta=beta,
        sampling=sampling,
        sampling_size=sampling_size,
        constraints=constraints,
        best_model_dir=best_model_dir,
        use_mlflow=use_mlflow,
        use_class_weights=use_class_weights,
        patience=patience,
        c_lr=c_lr,
        c_warmup_iters=c_warmup_iters,
        c_freq_increase=c_freq_increase,
        c_freq_increase_freq=c_freq_increase_freq,
        c_lr_decay=c_lr_decay,
        c_lr_decay_param=c_lr_decay_param,
        run_name=run_name,
    )
    temporal_yn.main(args)


@app.command()
def constraint_analysis(
    # Training parameters
    seed: int = typer.Option(42, help="Seed value used for experiment"),
    model: str = typer.Option("bert", help="Model used"),
    batch_size: int = typer.Option(8, help="Batch size for analysis"),
    # Data parameters
    data_path: str = typer.Option("data/", help="Path to the data folder"),
    # Model parameters
    dropout: bool = typer.Option(False, help="Enable dropout"),
    constraints: bool = typer.Option(False, help="Enable constraints"),
    # Training method parameters
    pmd: bool = typer.Option(False, help="Use Primal Dual method"),
    beta: float = typer.Option(1.0, help="Beta parameter for PMD"),
    sampling: bool = typer.Option(False, help="Use sampling loss"),
    sampling_size: int = typer.Option(4, help="Sampling size"),
    # Additional options
    cuda: int = typer.Option(0, help="CUDA device number (-1 for CPU)"),
    output_file: str = typer.Option(
        "final_chain_questions.json", help="Path to save the extracted chain questions as a JSON array"
    ),
):
    import argparse

    import tempQchain.constraint_analysis as constraint_analysis

    args = argparse.Namespace(
        seed=seed,
        model=model,
        batch_size=batch_size,
        data_path=data_path,
        dropout=dropout,
        constraints=constraints,
        pmd=pmd,
        beta=beta,
        sampling=sampling,
        sampling_size=sampling_size,
        cuda=cuda,
        output_file=output_file,
    )
    constraint_analysis.main(args)


@app.command()
def calculate_transitive_accuracy(
    file_path: str = typer.Option(..., help="Path to the constraint results file"),
):
    """Calculate transitive accuracy from constraint results file."""
    import tempQchain.transitive_accuracy as transitive_accuracy

    typer.echo(f"Analyzing transitive accuracy for: {file_path}")

    try:
        results = transitive_accuracy.calculate_transitive_accuracy(file_path)

        # Output results
        typer.echo("\nTransitive Constraint Analysis:")
        typer.echo(f"Total transitive batches: {results['total_transitive_batches']}")
        typer.echo(f"Correct conclusions: {results['correct_conclusions']}")
        typer.echo(f"Incorrect conclusions: {results['incorrect_conclusions']}")
        typer.echo(f"Accuracy: {results['accuracy']:.2%}")

        typer.echo("\nRule-by-Rule Statistics:")
        for rule_key, stats in results["rule_stats"].items():
            rule_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            typer.echo(f"  Rule {rule_key}: {stats['correct']}/{stats['total']} correct ({rule_accuracy:.2%})")

        typer.echo("✅ Transitive accuracy calculation completed successfully!")
    except Exception as e:
        typer.echo(f"❌ Error during transitive accuracy calculation: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def calculate_symmetry_accuracy(
    file_path: str = typer.Option(..., help="Path to the constraint results file"),
):
    """Calculate symmetry accuracy from constraint results file."""
    import tempQchain.symmetry_accuracy as symmetry_accuracy

    typer.echo(f"Analyzing symmetry accuracy for: {file_path}")

    try:
        results = symmetry_accuracy.calculate_symmetry_accuracy(file_path)

        # Output results
        typer.echo("\nSymmetry Constraint Analysis:")
        typer.echo(f"Total symmetry batches: {results['total_symmetry_batches']}")
        typer.echo(f"Correct conclusions: {results['correct_conclusions']}")
        typer.echo(f"Incorrect conclusions: {results['incorrect_conclusions']}")
        typer.echo(f"Accuracy: {results['accuracy']:.2%}")

        typer.echo("\nRule-by-Rule Statistics:")
        for rule_key, stats in results["rule_stats"].items():
            rule_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            typer.echo(f"  Rule {rule_key}: {stats['correct']}/{stats['total']} correct ({rule_accuracy:.2%})")

        typer.echo("✅ Symmetry accuracy calculation completed successfully!")
    except Exception as e:
        typer.echo(f"❌ Error during symmetry accuracy calculation: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def calculate_total_constraint_metrics(
    data_path: str = typer.Option(..., help="Path to the data folder containing constraint results files"),
):
    """Calculate symmetry and transitive accuracy for all files in a data folder."""
    from pathlib import Path

    import tempQchain.symmetry_accuracy as symmetry_accuracy
    import tempQchain.transitive_accuracy as transitive_accuracy

    data_dir = Path(data_path)

    if not data_dir.exists() or not data_dir.is_dir():
        typer.echo(f"❌ Data folder not found: {data_path}", err=True)
        raise typer.Exit(1)

    files = sorted(data_dir.iterdir())
    if not files:
        typer.echo(f"❌ No files found in directory: {data_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Analyzing constraint metrics for all files in: {data_path}\n")

    for file_path in files:
        if not file_path.is_file():
            continue

        typer.echo(f"{'='*60}")
        typer.echo(f"Processing file: {file_path.name}")
        typer.echo(f"{'='*60}")

        # Calculate symmetry accuracy
        typer.echo(f"\n[Symmetry Accuracy for {file_path.name}]")
        try:
            sym_results = symmetry_accuracy.calculate_symmetry_accuracy(str(file_path))
            typer.echo(f"Total symmetry batches: {sym_results['total_symmetry_batches']}")
            typer.echo(f"Correct conclusions: {sym_results['correct_conclusions']}")
            typer.echo(f"Incorrect conclusions: {sym_results['incorrect_conclusions']}")
            typer.echo(f"Accuracy: {sym_results['accuracy']:.2%}")

            typer.echo("\nRule-by-Rule Statistics (Symmetry):")
            for rule_key, stats in sym_results["rule_stats"].items():
                rule_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                typer.echo(f"  Rule {rule_key}: {stats['correct']}/{stats['total']} correct ({rule_accuracy:.2%})")
        except Exception as e:
            typer.echo(f"❌ Error calculating symmetry accuracy for {file_path.name}: {e}", err=True)

        # Calculate transitive accuracy
        typer.echo(f"\n[Transitive Accuracy for {file_path.name}]")
        try:
            trans_results = transitive_accuracy.calculate_transitive_accuracy(str(file_path))
            typer.echo(f"Total transitive batches: {trans_results['total_transitive_batches']}")
            typer.echo(f"Correct conclusions: {trans_results['correct_conclusions']}")
            typer.echo(f"Incorrect conclusions: {trans_results['incorrect_conclusions']}")
            typer.echo(f"Accuracy: {trans_results['accuracy']:.2%}")

            typer.echo("\nRule-by-Rule Statistics (Transitive):")
            for rule_key, stats in trans_results["rule_stats"].items():
                rule_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                typer.echo(f"  Rule {rule_key}: {stats['correct']}/{stats['total']} correct ({rule_accuracy:.2%})")
        except Exception as e:
            typer.echo(f"❌ Error calculating transitive accuracy for {file_path.name}: {e}", err=True)

        typer.echo()

    typer.echo("✅ Total constraint metrics calculation completed for all files!")


if __name__ == "__main__":
    app()
