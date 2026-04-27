"""Interactive marimo exploration app for the Hotelling market model.

Run with:
    marimo edit apps/explore_market.py

Or as a read-only web app:
    marimo run apps/explore_market.py
"""
import marimo as mo


app = mo.App(width="full")


@app.cell
def __(mo):
    mo.md(
        r"""
        # Hotelling Spatial Competition Explorer

        Interactively explore 2-D Hotelling spatial competition:
        - Adjust the **transport cost** $t$ and **logit scale** $\mu$
        - Watch how firm market areas (Voronoi cells) and equilibrium prices change
        - Compare Q-learning convergence against the Bertrand-Nash benchmark
        """
    )
    return


@app.cell
def __(mo):
    transport_cost = mo.ui.slider(
        0.1, 3.0, step=0.1, value=1.0, label="Transport cost $t$"
    )
    mu = mo.ui.slider(0.05, 1.0, step=0.05, value=0.25, label="Logit scale $\\mu$")
    m = mo.ui.slider(5, 25, step=1, value=15, label="Price levels $m$")
    mo.hstack([transport_cost, mu, m])
    return m, mu, transport_cost


@app.cell
def __(m, mo, mu, transport_cost):
    mo.md(
        f"""
        **Current parameters**

        | Parameter | Value |
        |-----------|-------|
        | Transport cost $t$ | {transport_cost.value:.2f} |
        | Logit scale $\\mu$ | {mu.value:.2f} |
        | Price levels $m$ | {m.value} |
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Market Structure

        With the current parameters, the symmetric duopoly Bertrand-Nash
        equilibrium price is approximately:

        $$p^* \approx c + t + \mu$$

        where $c$ is the marginal cost (set to 1.0 here).

        > **Note:** Full simulation requires running `make install-dev` and
        > configuring an LLM API key for the LLM agent comparison.
        """
    )
    return


@app.cell
def __(mo, transport_cost, mu):
    import math  # noqa: PLC0415

    # Approximate symmetric Nash price for logit spatial duopoly
    # (Anderson, de Palma, Thisse 1992, Ch.7)
    marginal_cost = 1.0
    approx_nash_price = marginal_cost + transport_cost.value + mu.value
    approx_monopoly_price = marginal_cost + transport_cost.value + 2 * mu.value

    mo.md(
        f"""
        ### Benchmark Prices (symmetric duopoly)

        | Equilibrium | Price |
        |-------------|-------|
        | Bertrand-Nash $p^*$ | {approx_nash_price:.3f} |
        | Joint Monopoly $p^m$ | {approx_monopoly_price:.3f} |
        | Price gap | {approx_monopoly_price - approx_nash_price:.3f} |
        """
    )
    return approx_monopoly_price, approx_nash_price, marginal_cost, math


if __name__ == "__main__":
    app.run()
