"""Printing the per-segment thermal load contribution table to the terminal."""

from __future__ import annotations


def print_pipe_thermal_table(network, decimals: int = 2):
    """
    Print thermal load fractions for each pipe segment.
    Fractions are assumed to already represent the fraction
    of the total thermal load.
    """

    headers = ["ID", "L [m]", "Heat [%]", "Cool [%]", "Sum [%]"]

    rows = []
    sum_heat = 0.0
    sum_cool = 0.0

    for seg in network.pipe_segments:

        heat_pct = 100 * getattr(seg, "F_heat", 0.0)
        cool_pct = 100 * getattr(seg, "F_cool", 0.0)

        total_pct = heat_pct + cool_pct

        sum_heat += heat_pct
        sum_cool += cool_pct

        rows.append([
            str(seg.id_),
            f"{seg.length:.0f}",
            f"{heat_pct:.{decimals}f}",
            f"{cool_pct:.{decimals}f}",
            f"{total_pct:.{decimals}f}",
        ])

    total_row = [
        "TOTAL",
        "",
        f"{sum_heat:.{decimals}f}",
        f"{sum_cool:.{decimals}f}",
        f"{(sum_heat+sum_cool):.{decimals}f}",
    ]

    widths = [max(len(x) for x in col) for col in zip(headers, *rows, total_row)]

    def fmt(row):
        return "  ".join(row[i].rjust(widths[i]) if i else row[i].ljust(widths[i]) for i in range(len(row)))

    print("\nThermal pipe contribution")
    print(fmt(headers))
    print("  ".join("-"*w for w in widths))

    for r in rows:
        print(fmt(r))

    print("  ".join("-"*w for w in widths))
    print(fmt(total_row))
