
from pathlib import Path

from .constants import Unit


def format_step(step: int) -> str:
    return format(step, "<10d")


def format_time(time_fs: float) -> str:
    return format(time_fs, "<10.2f")


def _ensure_output_path(filename: str) -> Path:
    path = Path(filename)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_energy(filename, energy_values, step, time_fs):
    path = _ensure_output_path(filename)
    if not energy_values:
        raise ValueError("energy_values must not be empty")
    with path.open("a+", encoding="utf-8") as fh:
        line = format_step(step) + format_time(time_fs)
        for value in energy_values:
            line += format(value, "<20.10f")
        fh.write(line + "\n")


def write_temperature(filename, temperature, step, time_fs):
    path = _ensure_output_path(filename)
    with path.open("a+", encoding="utf-8") as fh:
        fh.write(
            format_step(step)
            + format_time(time_fs)
            + format(temperature, "<20.10f")
            + "\n"
        )


def write_state(filename, state, step, time_fs):
    path = _ensure_output_path(filename)
    with path.open("a+", encoding="utf-8") as fh:
        fh.write(format_step(step) + format_time(time_fs) + format(state, "<5d") + "\n")


def write_coordinates(filename, symbols, matrix, step, time_fs):
    if len(symbols) != len(matrix):
        raise ValueError("Number of symbols does not match matrix length.")
    path = _ensure_output_path(filename)
    with path.open("a+", encoding="utf-8") as fh:
        fh.write(str(len(symbols)) + "\n")
        fh.write(f"Step={step}\tTime={time_fs:.2f}\n")
        for sym, coord in zip(symbols, matrix):
            fh.write(
                f"{sym:<2s}{coord[0]:>20.10f}{coord[1]:>20.10f}{coord[2]:>20.10f}\n"
            )
