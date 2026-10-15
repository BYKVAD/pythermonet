"""Printing the distribution pipe dimensioning table to the terminal."""

from __future__ import annotations

from pythermonet.dimensioning.hydraulic_result import HydraulicResult


def print_pipe_dimensioning_table(hydraulic: HydraulicResult):
    """
    Printer en tabel for det installerede trace-netværk.
    """

    N = len(hydraulic.diameters_outer)
    doCooling = hydraulic.reynolds_numbers_cooling is not None

    header_parts = [
        f"{'Trace':>6}",
        f"{'Mode':>10}",
        f"{'Do [mm]':>10}",
        f"{'Di [mm]':>10}",
        f"{'Re_heat':>12}",
        f"{'dp_heat [kPa]':>14}",
    ]

    if doCooling:
        header_parts += [
            f"{'Re_cool':>12}",
            f"{'dp_cool [kPa]':>14}",
        ]

    header = " ".join(header_parts)

    print("\nDistribution pipe dimensioning")
    print(header)
    print("-" * len(header))

    for i in range(N):
        row_parts = [
            f"{i:6d}",
            f"{str(hydraulic.governing_mode[i]):>10}",
            f"{hydraulic.diameters_outer[i] * 1000:10.1f}",
            f"{hydraulic.diameters_inner[i] * 1000:10.1f}",
            f"{hydraulic.reynolds_numbers_heating[i]:12.0f}",
            f"{hydraulic.pressure_losses_heating[i] / 1000:14.2f}",
        ]

        if doCooling:
            row_parts += [
                f"{hydraulic.reynolds_numbers_cooling[i]:12.0f}",
                f"{hydraulic.pressure_losses_cooling[i] / 1000:14.2f}",
            ]

        print(" ".join(row_parts))

    print()
