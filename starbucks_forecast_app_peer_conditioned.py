# --- Risk Summary ---
st.subheader("📊 Forecast Errors & Risk Flags")
st.dataframe(risk_flags.style.format({'Forecast': '${:,.0f}', 'Actual': '${:,.0f}', 'Error (%)': '{:.2f}%'}))

# --- Industry Benchmarking ---
industry_avg_growth = 0.04
last_actual = actual.loc['2023-01-01':].iloc[-1]
last_forecast = forecast_mean.loc['2023-01-01':].iloc[-1]
starbucks_growth = (last_forecast - last_actual) / last_actual

if last_forecast > last_actual:
    st.subheader("🏢 Industry Peer Benchmarking")
    st.write(f"📈 Starbucks Forecasted Growth: {starbucks_growth:.2%}")
    st.write(f"🏷️ Industry Average Growth: {industry_avg_growth:.2%}")

    if starbucks_growth > industry_avg_growth + 0.02:
        benchmark_flag = "⚠️ Starbucks forecast exceeds industry average by more than 2%. Review for potential overstatement."
    elif starbucks_growth < industry_avg_growth - 0.02:
        benchmark_flag = "ℹ️ Starbucks forecast is below industry average. This may indicate conservative assumptions or lower performance expectations."
    else:
        benchmark_flag = "✅ Starbucks forecast is aligned with industry averages."

    st.markdown(f"**{benchmark_flag}**")

# --- Summary ---
high_risk_2023 = risk_flags.loc['2023-01-01':]
high_risk_count = (high_risk_2023['Flag'] == '🚨 High Risk').sum()
avg_error_2023 = high_risk_2023['Error (%)'].mean()

if high_risk_count == 0:
    summary_text = f"""### 📌 Summary for Audit Committee
No high-risk quarters were identified in 2023. The ARIMAX forecast using CPI ({cpi_input:.2f}), avg ticket ({avg_ticket_input:.2f}), and transactions ({txn_input}) closely aligns with reported revenue.

This consistency suggests that Starbucks' reported revenue is well-supported by key economic and operational inputs. The regression model further confirms this by showing minimal variance between expected and actual revenues across all quarters. The observed alignment between revenue growth and input growth patterns strengthens the case for reliability in reported figures.

Based on this evidence, there is no indication of revenue overstatement in the reviewed period, and the financial reporting appears to reflect the underlying business conditions accurately.
"""
elif high_risk_count == 1:
    summary_text = f"""### 📌 Summary for Audit Committee
One high-risk quarter was flagged in 2023 where actual revenue materially exceeded forecasted values. The model, based on CPI ({cpi_input:.2f}), avg ticket ({avg_ticket_input:.2f}), and transactions ({txn_input}), generally aligns well otherwise.

The presence of a single significant deviation warrants targeted scrutiny. Auditors should consider whether temporary operational or macroeconomic anomalies might explain the gap. If no external justification is found, this deviation could indicate a timing or estimation error in revenue recognition.

While not indicative of widespread overstatement, this anomaly suggests the need for further documentation and review of revenue policies for the flagged quarter.
"""
else:
    summary_text = f"""### 📌 Summary for Audit Committee
{high_risk_count} high-risk quarters were flagged in 2023 with an average deviation of {avg_error_2023:.2f}%. Forecasts based on CPI ({cpi_input:.2f}), avg ticket ({avg_ticket_input:.2f}), and transactions ({txn_input}) showed persistent misalignment with reported revenue.

Such repeated deviations raise substantial concern about potential overstatement of revenue. The regression model also reflects a breakdown in the relationship between operational drivers and reported figures, especially where revenue growth far exceeds changes in underlying inputs.

Auditors should investigate these flagged quarters thoroughly, focusing on revenue recognition timing, estimates, and the validity of recorded transactions. Evidence of revenue recognition that outpaces business activity may warrant extended testing and disclosure consideration.
"""

st.markdown(summary_text)
