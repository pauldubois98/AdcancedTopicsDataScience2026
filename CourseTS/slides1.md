---
title: |
  Session 5 — Part 1\
  Time Series: Structure
subtitle: "Advanced Data Science · NYU Paris · 2026"
author: "Paul Dubois"
date: "Week 5 — Lecture"
---

## Main question
> The rows of our table are no longer exchangeable.
>
> What changes when the rows are ordered?

## Time series

![](img/ts_zoo.png)

## The order carries the signal 

![](img/iid_broken.png)

## Series decomposition

![](img/components.png)

## Additive or multiplicative

![](img/eq_decomposition.png)

::: notes
almost every tool only implements the additive case
:::

## Additive, multiplicative, and the log

![](img/add_vs_mult.png)

::: notes
Not a seed: the log of a LINEAR trend is concave. The multiplicative twin now has
an exponential trend, so its log is straight.
:::

## Estimating the trend

![](img/moving_average.png)

## The ends of a moving average

![](img/ma_edges.png)

::: notes
NaN is the honest answer, and the missing points are the most recent ones.
:::

## Classical decomposition, step by step

![](img/classical_decomp.png)

::: notes
The second term in step 3 is what makes the seasonal component sum to zero over a
cycle. Without it the season would carry part of the level.
:::

## Loess = LOcally Estimated Scatterplot Smoothing 

![](img/loess_build.png)

::: notes
Weight the neighbours, fit a straight line, keep the single value above $t_0$, then move $t_0$ along.
:::

## Loess: the weights and the span

![](img/loess_span.png)

::: notes
A moving average is the special case: flat weights, and a local constant instead of a local line.
:::

## Seasonal Trend decomposition with Loess

![](img/stl.png)

::: notes
The inner loop alternates between estimating the season and estimating the trend.
The outer loop re-runs it with outliers down-weighted.
:::

## STL inner loop

![](img/stl_inner.png)


## STL: the outer loop

![](img/stl_outer.png)

::: notes
One planted outlier. Its residual is nine times the typical one, so the bisquare
weight sends it to almost zero and the next pass fits as if it were not there.

Set $n_o = 0$ when the data is clean: each robustness pass costs a full inner loop.
:::

# Stationary Time Series

## Stationary Time Series

![](img/eq_stationarity.png)

## Stationary Time Series

![](img/stationary_zoo.png)

::: notes
The cycle is the one they get wrong: it wanders, it looks like it trends in
places, and it is stationary.
:::

## Benefits of Stationary Time Series

![](img/why_stationarity.png)

::: notes
So stationarity is the assumption that makes one long series a usable substitute for many short ones.
:::

## White noise

![](img/white_noise.png)

::: notes
The reference process: constant mean, constant variance, zero covariance at every lag.
Stationary by inspection, and nothing in it to predict.
:::


## Auto Regressive process AR

![](img/ar_intro.png)

::: notes
Keep a fraction $\phi$ of where you were, add a fresh shock. A regression whose
only feature is the series itself, one step back.
:::

## φ is a memory dial

![](img/ar_memory.png)

::: notes
A shock is worth $\phi^h$ after $h$ steps, so $\phi$ decides how long the process remembers.
:::

## Random walk

![](img/random_walk_intro.png)

::: notes
$\phi = 1$: nothing decays, so the level is the running total of every shock ever
received. That is the bridge to the next slide.
:::

## Variance of random walks & auto regression

![](img/random_walk.png)

## φ values for auto regression

![](img/unit_root.png)

::: notes
Read the y scales across the four panels: the last one is three orders of
magnitude bigger than the others.
:::

## Auto Regressive process of higher orders AR(p)

![](img/ar_p.png)

::: notes
The same idea with $p$ lags. Still a linear regression, but stationarity is no
longer a condition you can read off a single number.
:::

## The backshift operator

![](img/backshift.png)

::: notes
$B$ shifts one step back and composes like a number, which turns the recursion
into a single polynomial applied to the series.
:::

## The characteristic equation

![](img/char_eq.png)

::: notes
Replace $B$ by a complex number $z$ and solve. Every root outside the unit circle
means stationary. For AR(1) it is $|\phi| < 1$ written differently.
:::

## Root cases

![](img/root_cases.png)

::: notes
Three cases, three behaviours. Note the log scale on the explosive panel: at
$\phi = 2$ the series passes $10^{45}$ within 150 steps.
:::

## Moving average process MA(q)

![](img/ma_intro.png)

::: notes
The other way to build a process out of white noise, and the last building block
we need.

AR fed the shocks through a recursion, so the memory decays forever. MA just adds
up the last q shocks and stops. That gives a memory of exactly q steps: $y_t$ and
$y_{t-q-1}$ have no shock in common at all, so their covariance is exactly zero
and the ACF cuts off. That cut-off is the property the identification table later
in the session is built on.

Two things to insist on. First, the inputs are invisible — you observe $y_t$, you
never observe $\varepsilon_t$. That is why fitting an MA needs maximum likelihood
rather than least squares: you cannot regress on something you have not measured.
Second, it is always stationary, with no conditions on θ: a finite sum of
finite-variance terms always has finite variance.

Now the name, because it is genuinely confusing and they will trip on it. This is
NOT the moving average from the decomposition section. That one averages OBSERVED
values to pull a trend out of data; this one is a MODEL of how the data arose, and
what it averages cannot be seen. The arithmetic is the same shape, which is why the
name got reused — run a moving-average filter over white noise and what comes out
IS an MA process. The two-point average of white noise has ρ(1) = 0.5, which is
exactly the MA(1) with θ = 1.

That connection is worth one sentence and no more; the important thing is that they
do not read MA(2) as "a two-point moving average".
:::

## Spurious regression

![](img/spurious_regression.png)

::: notes
Left: two series. I generated them with two separate random number streams.
There is no mechanism connecting them, none, (I wrote the code).

Middle: scatter the levels against each other and get a correlation around 0.9.
A regression would get a large coefficient, a tiny p-value, and a high R².
Every diagnostic a naive analyst checks would be green.

The reason is that both series have a slowly moving level, so both spend long stretches high and long stretches low.
Any two series with wandering levels will appear to move together over a finite sample.

Right: difference both and scatter again.
The correlation collapses to nearly zero, which is the truth.

The rule: never regress one non-stationary series on another.
Difference first, or use the proper apparatus for it, which is cointegration and is beyond today.
:::

# Stationarity tests

## Dickey-Fuller intuition

![](img/df_intuition.png)

::: notes
The whole test in one sentence: does the LEVEL tell you anything about the next
CHANGE?

If the series is stationary it is pulled back towards its mean, so a high value
tends to be followed by a fall and a low value by a rise. Regress the change on
the level and the coefficient comes out negative.

If it is a random walk there is no mean to return to. Where you are now says
nothing about which way you go next, so that coefficient is zero.

The two scatter plots are that sentence made visible, on real draws. Left: slope
−0.400, and the theory says it should be φ − 1 = −0.4. Right: slope −0.000.

So the test is just: fit that line, and ask whether the slope is meaningfully
negative.
:::

## The Dickey-Fuller test

![](img/df_test.png)

::: notes
Left, the algebra, and it is two lines. Start from $y_t = \phi y_{t-1} +
\varepsilon_t$, subtract $y_{t-1}$ from both sides, and the equation becomes
$\Delta y_t = (\phi - 1) y_{t-1} + \varepsilon_t$. Write γ = φ − 1 and the unit
root question — is φ = 1? — has become an ordinary question about one regression
coefficient: is γ = 0?

It is one-sided. γ > 0 would mean explosive, which we are not entertaining, so
only large NEGATIVE values count against the null.

Middle: three versions, and the choice is yours to make. No constant at all, a
constant, or a constant and a trend. That last one is testing against
TREND-stationarity, which is a different alternative. Each has its own table.

Right, and this is the part that surprises people: you cannot look the statistic
up in a $t$ table. Under the null the regressor $y_{t-1}$ is itself a random walk,
which breaks the usual asymptotics. The distribution is shifted well to the left —
I simulated it from 4000 random walks — and the 5% critical value is about −2.9
rather than −1.96. Using the $t$ table would reject far too often.
:::

## From DF to ADF

![](img/adf_test.png)

::: notes
The chain of reasoning, left to right.

The plain test assumes its errors are white noise. A real series is rarely an
AR(1), so its increments are usually correlated, and then that assumption is
false. Why does that matter? Because se$(\hat\gamma)$ — the number we divided by
on the last slide — is computed FROM that assumption. A wrong se gives a wrong
$t$, and a wrong $t$ gives a wrong decision.

The middle panel measures how wrong, and this is the slide's point. Every series
in it genuinely HAS a unit root, so the null is true and the test should reject 5%
of the time. With iid increments the plain test does: 4%. Make the increments
correlated and it rejects 34%, then 82%. You would conclude "stationary" on four
out of five series that are nothing of the sort.

The green bars are the same series run through ADF with 8 lags: 3% every time.

Right: the fix and its limits. The δ terms are the only new thing, γ still answers
the same question, and the null distribution is UNCHANGED — the same table we
simulated two slides ago. Adding lags repairs the standard error; it does not
change what you compare against.
:::

## ADF, step by step

![](img/adf_recipe.png)

::: notes
Six steps on the left, and the right half exists to stop step 1 being read as
paperwork.

The series top right is a straight line plus noise. It is not stationary — the
mean climbs — but it IS trend-stationary: take the line out and what is left is
white noise.

Now run the test twice on that one series. Under "c" the statistic is −0.79
against a critical value of −2.87, so you do not reject, and you would report "it
has a unit root". Under "ct" the same data gives −5.06 against −3.42, so you
reject and report "trend-stationary".

Same series, same code, opposite conclusions. The choice in step 1 is not a
setting, it is the question you are asking: "c" asks whether the series returns to
a constant, "ct" asks whether it returns to a line. Decide which one you meant
BEFORE you look at the p-value.
:::

## KPSS intuition

![](img/kpss_intuition.png)

::: notes
The mirror of the Dickey-Fuller intuition slide, and the same trick: ask one
question of the data and let the test be the answer.

A stationary series is pulled back to its mean, so its deviations from that mean
have to keep changing sign — it cannot stay above for long without coming back.
Add those deviations up as you go and the running total keeps getting cancelled.
Left column: 111 sign changes, and the running total never gets past 20.

A series with a random walk in it has no mean to return to. Its deviations keep
the same sign for long stretches, so the running total accumulates instead of
cancelling. Right column: only 22 sign changes, and the total reaches 1474.

That is the entire test. LM is essentially the area under the squared running
total, scaled so it does not grow with n: 0.05 against 3.46 here.

Say the direction out loud, because it is the opposite of ADF: a BIG statistic
means the sums ran away, which means NOT stationary. KPSS rejects in the upper
tail.
:::

## Kwiatkowski Phillips Schmidt Shin (KPSS) test

![](img/kpss_recipe.png)

::: notes
Same five series, so the two tables can be read against each other.

Note the direction: KPSS rejects in the UPPER tail, because a large LM means the
partial sums wander. And row 3 is wrong here too, in the opposite direction —
φ = 0.95 gets flagged as non-stationary when it is not.
:::

## ADF and KPSS

![](img/adf_kpss.png)

::: notes
Two tests, and the thing students get wrong is that their nulls are opposite.

ADF, Augmented Dickey-Fuller: the null is that there IS a unit root, so
non-stationary. Rejecting is the good outcome. KPSS: the null is that the series IS
stationary. Rejecting is the bad outcome.

So a small p-value is good news from ADF and bad news from KPSS. Say that twice;
it is the most common error in this part of the course.

Right, the 2×2, which is how you should actually use them. Top left, ADF rejects
and KPSS does not: both tests agree on stationary, proceed. Top right, the mirror
image: both agree on non-stationary, difference it. Bottom left, both reject:
usually a variance problem rather than a mean problem, so look at a transform, or
a structural break. Bottom right, neither rejects: you do not have enough data to
distinguish, and no amount of testing will fix that.

Then the caveat in red, which applies to every test in the session. Failing to
reject is not evidence for the null. And these tests have famously poor power
against φ = 0.95, which we just looked at. Plot the series first. Your eyes are a
better instrument than the p-value here, and the tests are there to check your
eyes, not to replace them.
:::

# Making a time series stationary

## Differencing

![](img/differencing.png)

::: notes
The standard tool, and there are exactly two kinds.

Top left: the raw series, trend and seasonality both present.

Top right: the first difference, y_t − y_{t−1}.
The trend is gone: a linear trend differenced once becomes a constant
but the twelve-step oscillation is still plainly there.

Bottom left: the seasonal difference, y_t − y_{t−12}.
The seasonality is gone.
Comparing each point to the same phase one year earlier.
This is the "year over year change".

Bottom right: both, applied one after the other.
Noise around zero, and that is what the ARMA machinery wants as input.

Two practical notes.
The order does not matter, ∆∆₁₂ = ∆₁₂∆.
:::

## Over differencing

![](img/over_differencing.png)

::: notes
The counterweight to the previous slide.

Start from a series that is ALREADY stationary — an AR(1) with φ = 0.2, top left, variance 1.11.

Difference it once: the variance goes UP, to 1.74.
Difference it twice: up again, to 4.90.
Differencing amplifies high-frequency noise.

The factor is exact: Var(∆x) = 2(1 − φ) Var(x).
So it only inflates the variance when φ < 0.5.
Above that the series is smooth enough that differencing SHRINKS it — which is
the whole point of differencing a trending series in the first place.
The cost that is always present is the one in the bottom row.

Now the bottom row, which is the diagnostic.
The ACF of the over-differenced series has a large NEGATIVE spike at lag 1 that was not there before.

That is not a discovery about the data, it is an artefact.

Differencing white noise produces a process with lag-1 autocorrelation of exactly −0.5, an MA(1) unit root.
It is mathematically impossible to invert.

Difference the minimum number of times that makes the series stationary.
:::

## Box-Cox: fixing the variance

![](img/box_cox_why.png)

::: notes
The goal is a series whose VARIANCE no longer depends on its LEVEL. Measure the
spread early and late: 11 against 54 before the transform, 1.28 against 1.21
after it.

That is what the rest of the session assumes — constant variance is one of the
three stationarity conditions, an additive decomposition needs a constant seasonal
swing, and a prediction interval reuses one error size at every horizon.

Differencing fixes the mean; this fixes the spread.
:::

## Box-Cox: what λ does

![](img/box_cox_lambda.png)

::: notes
λ is a dial saying how hard to compress the large values. λ = 1 leaves the series
alone, λ = 0 is the log, and anything below 1 pulls the big values in, which is
exactly what a fanning spread needs.
:::

## Box-Cox: choosing λ

![](img/box_cox_choice.png)

::: notes
Try a λ, transform, fit the model you actually intend to use, record how well it
fits, correct for the change of scale, repeat. The best λ is the maximum of that
curve.

Step 5 is the one people drop. Without the Jacobian term $(λ-1)\sum_t \log y_t$
you would always choose the λ that squashes the series flattest, because a smaller
series trivially has a smaller variance.

Here the maximum is at −0.02 with a 95% interval of [−0.09, 0.04], so report
λ = 0 and take the log. The series was built as $e^{\,0.013t}$ times lognormal
noise, so that is the right answer.
:::

# Autocorrelation

## Lag

![](img/lag_def.png)

::: notes
$y_{t-h}$ is the value $h$ steps earlier. Shifting costs you the first $h$ rows,
which is why a lag-12 feature throws away a year.
:::

## Correlation with the past

![](img/eq_acf.png)

::: notes
Three lines, building up.

The autocovariance γ(h) is the covariance between the series and itself h steps
later. Point at the notation and make the stationarity link explicit: we can write
γ(h) with one argument only BECAUSE we assumed stationarity. Without it we would
need γ(t, h), a different number for every starting point, and it would be
unestimable.

Dividing by γ(0), which is the variance, gives ρ(h), the autocorrelation. Unit-free,
between −1 and 1, and ρ(0) = 1 always.

The estimator at the bottom has one oddity worth naming, because they will notice
it and assume it is a typo. The numerator has n − h terms; the denominator has n.
This is the biased estimator, and it is the one every package uses. The reason is
that it guarantees the resulting sequence is a valid autocorrelation function —
positive semi-definite — which the unbiased version does not. The cost is that
estimates at large h are shrunk towards zero, which is a feature, since those are
the noisiest ones anyway.
:::

## Auto Correlation

![](img/acf_def.png)

::: notes
The correlation of the series with its own lagged copy. One number per lag; the
whole set is the ACF. We come back to the estimator and its pitfalls later.
:::


## Correlogram

![](img/acf_build.png)

::: notes
This slide exists to stop the ACF being a magic plot.

First three panels: take the series, pair each point with the point one step later,
and scatter. That is a scatterplot like any other, and it has a correlation
coefficient, 0.74 here. Do the same at lag 2, and at lag 8, and the cloud gets
rounder as the correlation falls.

Fourth panel: put those correlations on one axis, one stem per lag. That is the
correlogram. Nothing else is going on.

Two reading conventions to state once. Lag 0 is always 1 and is conventionally
hidden — I drop it in all of these plots. The shaded band is the region where you
would expect an estimate to land if the truth were zero; we come back to it in two
slides.

Make sure the scatterplot picture sticks, because it makes the next two slides
readable without any new machinery.
:::

## Shapes worth recognising

![](img/acf_zoo.png)

::: notes
The vocabulary slide. Four series on top, their correlograms below, and the goal
is recognition on sight.

White noise: everything inside the band. There is nothing to model.

Trend: slow decay, everything positive out to high lags. Say why in one sentence:
if the series is drifting upwards, then any two points close in time are both above
the overall mean or both below it, so every short-lag correlation is positive. A
slowly decaying, all-positive ACF is a signature of a trend, not of long memory,
and the fix is differencing.

Seasonal: spikes at 12, 24, 36. The period is readable straight off the plot, which
is the fastest way to find m when nobody told you what it is.

AR(1) with φ = 0.8: geometric decay, ρ(h) = 0.8^h. Compare with the trend panel and
note they look similar at short lags — which is exactly why "does this series have
a trend or is it a highly autocorrelated stationary process" is a genuinely hard
question, and why unit root tests exist.

Real data usually shows two or three of these at once. Trend plus seasonality is
the standard case, and you difference both away before reading anything else.
:::

## The band

![](img/acf_bands.png)

::: notes
A short slide about not over-reading a plot they are about to use constantly.

Left: pure white noise, 200 points, and two or three lags poke outside the band.
Nothing is happening in that data — I generated it with a normal random number
generator.

Middle: where the band comes from. Under the null that the series is white noise,
each sample autocorrelation is approximately normal with variance 1/n, so the 95%
band is ±1.96/√n. Note the consequence at the bottom: the band shrinks with n, so
with a million observations a correlation of 0.003 is "significant". Significant and
large are different things, and on this plot you read the HEIGHT.

Right, the two traps. The band is a per-lag 5% test, so across 40 lags you expect
about two excursions by chance — exactly what the left panel shows. And the band's
derivation assumes white noise, so once lag 1 is large the band is not valid for
the later lags anyway.

The fix for both is a portmanteau test, which tests the whole correlogram at once.
That is Ljung-Box, and we get to it at the end of this section.
:::

## ACF vs PACF

![](img/pacf_idea.png)

::: notes
The PACF is the one students nod along to and cannot explain afterwards. The left
panel is the explanation.

An AR(1): y_t depends on y_{t−1} and on nothing else. Look at the diagram — there
is a blue arrow from t−2 to t−1, and from t−1 to t, and NO arrow from t−2 to t. By
construction.

But ρ(2) is not zero, it is about 0.64. The correlation between y_t and y_{t−2} is
real; it is just entirely inherited. They are correlated because both are tied to
y_{t−1}.

The PACF asks a different question: what is the correlation between y_t and y_{t−2}
AFTER removing everything explained by the points in between? Formally, regress
both on y_{t−1}, and correlate the residuals. For this process the answer is
exactly zero, and the third panel shows that.

So: ACF measures total association, direct plus inherited. PACF measures direct
association only. Anyone who has met partial correlation or "controlling for a
confounder" in a regression course has met this idea already — say that, it lands.

One implementation note if anyone asks: you do not actually run those regressions.
The Durbin-Levinson recursion computes the whole PACF from the ACF, and that is
what my helper does.
:::

## PACF

![](img/pacf_def.png)

::: notes
The definition in one line: regress $y_t$ on its first $h$ lags and keep the LAST
coefficient, $\phi_{hh}$. Everything else in that regression is there only to be
controlled for.

The middle panel is the proof, not an illustration: I fit AR(1) through AR(5) to
the same series and print the last coefficient beside the value my PACF routine
returns. They agree to three decimals.

At $h = 1$ there is nothing in between, so $\phi_{11} = \rho(1)$ — the ACF and the
PACF always start at the same number.

And the cut-off falls straight out: an AR(2) needs no third lag, so once lags 1
and 2 are in the model the coefficient on the third is zero. That is the property
the identification table on the next slide is built on.
:::

## The identification table

![](img/acf_pacf_signature.png)

::: notes
The payoff of the last two slides, and the reason Box and Jenkins built the method
this way.

Top row: an AR(2). Its ACF tails off gradually — there is no lag after which it is
zero. Its PACF has two significant spikes and then stops dead. The order is
readable: p = 2.

Bottom row: an MA(2). Exactly the reverse. The ACF cuts off after lag 2 — and that
is mechanical, since y_t and y_{t−3} share no shocks at all, so their covariance is
exactly zero. The PACF tails off.

Right, the table. AR: ACF tails, PACF cuts off at p. MA: ACF cuts off at q, PACF
tails. ARMA: both tail off, and neither plot gives you the order, which is why in
practice people fit a small grid and compare AIC.

Then the red line, which matters more than the table. These are 600-point
simulations from the exact model. Real data gives you 80 points from a process that
is not in the family, and the plots are ambiguous. Use the table to pick two or
three candidates. Use a backtest to choose between them.
:::


## Ljung-Box intuition

![](img/ljung_box_intuition.png)

::: notes
The third intuition slide in the same shape as the other two: ask one question,
and let the statistic be the answer.

The problem first. A correlogram with 24 lags is 24 separate judgements, and you
have to come out of it with ONE decision: is this model finished or not? Counting
how many bars poke out of the band does not work, because a few always do by
chance — that is what a 95% band means.

Look at the two top panels and try to call it by eye. White noise on the left has
2 bars outside; the AR(1) on the right has 5. Nothing about "2 versus 5" is
convincing, and on a different draw the numbers would swap around.

So stop judging bars. Square every one of them and add them up, weighted so that
each contributes on the same scale. One big spike makes the total large; so do
many small ones that all point the same way. That total is Q.

The bottom row is that sum being accumulated lag by lag. The white-noise total
drifts up to 25 and stays under the 5% threshold of 36.4. The AR(1) total crosses
it around lag 8 and finishes at 55. Same picture, one number, one decision.

Note what the left-hand series shows: two bars outside the band and yet Q says
there is nothing there. That is the slide's point.
:::

## Ljung-Box test

![](img/ljung_box_null.png)

::: notes
The null is a joint one: every autocorrelation from lag 1 to lag m is zero, all at
once. That is the point of a portmanteau test — reading 40 individual bands gives
you about two false alarms by construction, and this gives you one decision.

The middle panel builds the statistic rather than asserting it. Under the null each
sample autocorrelation has variance roughly (n−h)/(n(n+2)); standardise, square, and
add m of them, and a sum of m squared standard normals is a chi-squared with m
degrees of freedom. The n(n+2)/(n−h) weight is just that standardisation.

The right panel checks it: 4000 white-noise series, and the histogram of Q sits on
the chi-squared density. Slightly heavy in the far tail, which is the known
small-sample behaviour.

Box-Pierce is the same idea with the weight dropped; Ljung and Box added it because
the approximation was poor at the sample sizes people actually have.
:::

## Ljung-Box steps

![](img/ljung_box_recipe.png)

::: notes
Six steps. Step 1 is the one people get wrong: this goes on RESIDUALS, never on the
raw series — on raw data it will reject for any series with a trend, which tells
you nothing.

Step 5 is the other one. The m − k correction subtracts the ARMA parameters, p + q.
A regression on a trend and seasonal dummies fits no ARMA terms at all, so k = 0
here, and df = 24 in both rows.

The two rows are the same series under two models: forget the seasonality and Q is
1406; model it and Q is 21 against 24 degrees of freedom, p = 0.63. That is what
"no evidence against white" looks like.
:::
