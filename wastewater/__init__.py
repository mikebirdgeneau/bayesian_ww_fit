import datetime
import pickle

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pylab as plt
from tensorflow_probability import sts


class WastewaterModel:
    # Define variables:
    location = None
    data = None
    exclude_last_n_days = None
    df_merged = None
    masked_series = None
    model = None
    samples = None
    kernel_results = None
    forecast = None
    components = None
    one_step = None

    # Define the model components:
    def __init__(self, location, exclude_last_n_days=0):
        self.location = location
        self.exclude_last_n_days = exclude_last_n_days
        self.fetch_data()
        self.preprocess_data(exclude_last_n_days=exclude_last_n_days)
        self.create_model()

    # Save current object (self) to disk:
    def save(self, filepath='output/model.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def fetch_data(self):
        # Fetch latest data:
        url = 'https://chi-covid-data.pages.dev/aplWasteWaterAbData.json'
        r = requests.get(url)
        data = r.json()
        self.data = data

    def preprocess_data(self, exclude_last_n_days=0):
        # Location Data
        try:
            df = pd.DataFrame(self.data.get('data').get(self.location))
        except:
            raise Exception("Location {} not found in data.".format(self.location))
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'avg']]
        df = df.rename(columns={'avg': 'value'})

        # Exclude last n days (holdback):
        if exclude_last_n_days > 0:
            df = df[:-exclude_last_n_days]

        if(len(df) == 0):
            raise Exception("No data found for location {}, or holdback period exceeds available data.".format(self.location))

        # Handle irregularly sampled data:
        min_date, max_date = df['date'].min(), df['date'].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        df_full = pd.DataFrame(date_range, columns=['date'])
        df_merged = df_full.merge(df, on='date', how='left')

        # Create mask and fill missing values
        df_merged['mask'] = ~df_merged['value'].isna()
        self.df_merged = df_merged

        # Create masked time series, all values should be float64:
        masked_series = tfp.sts.MaskedTimeSeries(
            time_series=np.log(df_merged['value']).to_numpy(),
            is_missing=~df_merged['mask'].to_numpy()
        )

        self.masked_series = masked_series

    def create_model(self):

        semi_local_linear_trend = sts.SemiLocalLinearTrend(
            observed_time_series=self.masked_series,
            name='semi_local_linear_trend'
        )
        autoregressive = sts.Autoregressive(
            order=1,  # or 2, based on data analysis
            observed_time_series=self.masked_series,
            name='autoregressive'
        )
        # Sum the components to create the model:
        model = sts.Sum(
            components=[semi_local_linear_trend, autoregressive],
            observed_time_series=self.masked_series
        )
        self.model = model

    def fit_model(self, step_size=0.1, num_leapfrog_steps=10, num_warmup_steps=50, num_results=100, seed=42, parallel_iterations=10):

        transformed_hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=self.model.joint_distribution(self.masked_series).log_prob,
                    step_size=step_size,
                    num_leapfrog_steps=num_leapfrog_steps,
                    state_gradients_are_stopped=True),
                num_adaptation_steps=int(0.8 * num_warmup_steps)),
            bijector=[param.bijector for param in self.model.parameters])

        # Initialize from a Uniform[-2, 2] distribution in unconstrained space.
        initial_state = [tfp.sts.sample_uniform_initial_state(
            param, return_constrained=True) for param in self.model.parameters]

        # Define the sampling as a tf.function:forecast
        @tf.function
        def run_chain_fn():
            chain_result = tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_warmup_steps,
                current_state=initial_state,
                kernel=transformed_hmc_kernel,
                parallel_iterations=parallel_iterations,
                seed=seed,
                trace_fn=(lambda current_state, kernel_results: kernel_results))
            return chain_result

        # Run HMC sampling:
        samples, kernel_results = run_chain_fn()

        self.samples = samples
        self.kernel_results = kernel_results

    def plot_traces(self):
        for (param, param_draws) in zip(self.model.parameters, self.samples):
            if param.prior.event_shape.ndims > 0:
                print("Only plotting traces for scalar parameters, skipping {}".format(
                    param.name))
                continue
            plt.figure(figsize=(10, 4))
            plt.title(param.name)
            plt.plot(param_draws.numpy())
            plt.ylabel(param.name)
            plt.xlabel("HMC step")
            plt.savefig("output/trace_{}_{}.png".format(self.location, param.name.replace('/', '_')))

    def plot_hist(self):
        # Plot the marginal posteriors for each parameter:
        for (param, param_draws) in zip(self.model.parameters, self.samples):
            if param.prior.event_shape.ndims > 0:
                print("Skipping plot for {}".format(param.name))
                continue
            plt.figure(figsize=(6, 4))
            plt.title(param.name)
            plt.hist(param_draws.numpy(), bins=50)
            plt.xlabel("HMC step")
            plt.savefig("output/hist_{}_{}.png".format(self.location, param.name.replace('/', '_')))

    def combine_samples(self):
        # Combine samples from multiple chains:
        combined_samples = [np.reshape(param_draws[-1] + list(param_draws.shape[2:]))
                            for param_draws in self.samples]
        return combined_samples

    def component_distributions(self):
        # Get the posterior distributions of each component:
        component_dists = sts.decompose_by_component(
            self.model,
            observed_time_series=self.masked_series,
            parameter_samples=self.samples)

        components = [component_dists[component_name] for component_name in component_dists.keys()]

        self.components = components


    def predict_one_step(self):

# Get the posterior predictive distribution of each component:
        one_step_dist = sts.one_step_predictive(
            self.model,
            observed_time_series=self.masked_series,
            parameter_samples=self.samples)

        num_samples = 1000
        one_step_mean = np.exp(one_step_dist.mean())
        one_step_scale = np.exp(one_step_dist.stddev())
        one_step_samples = np.exp(one_step_dist.sample(num_samples))
        # Get forecast band (80% confidence interval):
        one_step_p10 = np.percentile(one_step_samples, 10, axis=0, method='interpolated_inverted_cdf')
        one_step_p90 = np.percentile(one_step_samples, 90, axis=0, method='interpolated_inverted_cdf')
        one_step_p05 = np.percentile(one_step_samples, 5, axis=0, method='interpolated_inverted_cdf')
        one_step_p95 = np.percentile(one_step_samples, 95, axis=0, method='interpolated_inverted_cdf')
        one_step_p25 = np.percentile(one_step_samples, 25, axis=0, method='interpolated_inverted_cdf')
        one_step_p75 = np.percentile(one_step_samples, 75, axis=0, method='interpolated_inverted_cdf')
        one_step_median = np.percentile(one_step_samples, 50, axis=0, method='interpolated_inverted_cdf')

        one_step_result = {
            'one_step_mean': one_step_mean,
            'one_step_scale': one_step_scale,
            'one_step_samples': one_step_samples,
            'one_step_median': one_step_median,
            'one_step_p10': one_step_p10,
            'one_step_p90': one_step_p90,
            'one_step_p05': one_step_p05,
            'one_step_p95': one_step_p95,
            'one_step_p25': one_step_p25,
            'one_step_p75': one_step_p75,
        }

        self.one_step = one_step_result



    def predict(self, days_to_forecast=None):

        if days_to_forecast is None:
            days_to_forecast = (datetime.datetime.today() - self.df_merged['date'].max()).days + 1

        forecast_dist = tfp.sts.forecast(
            model=self.model,
            observed_time_series=self.masked_series,
            parameter_samples=self.samples,
            num_steps_forecast=days_to_forecast)

        num_samples = 1000
        forecast_mean = np.exp(forecast_dist.mean()[..., 0])
        forecast_scale = np.exp(forecast_dist.stddev()[..., 0])
        forecast_samples = np.exp(forecast_dist.sample(num_samples)[..., 0])
        # Get forecast band (80% confidence interval):
        forecast_p10 = np.percentile(forecast_samples, 10, axis=0, method='interpolated_inverted_cdf')
        forecast_p90 = np.percentile(forecast_samples, 90, axis=0, method='interpolated_inverted_cdf')
        forecast_p05 = np.percentile(forecast_samples, 5, axis=0, method='interpolated_inverted_cdf')
        forecast_p95 = np.percentile(forecast_samples, 95, axis=0, method='interpolated_inverted_cdf')
        forecast_p25 = np.percentile(forecast_samples, 25, axis=0, method='interpolated_inverted_cdf')
        forecast_p75 = np.percentile(forecast_samples, 75, axis=0, method='interpolated_inverted_cdf')
        forecast_median = np.percentile(forecast_samples, 50, axis=0, method='interpolated_inverted_cdf')

        forecast_result = {
            'forecast_mean': forecast_mean,
            'forecast_scale': forecast_scale,
            'forecast_samples': forecast_samples,
            'forecast_median': forecast_median,
            'forecast_p10': forecast_p10,
            'forecast_p90': forecast_p90,
            'forecast_p05': forecast_p05,
            'forecast_p95': forecast_p95,
            'forecast_p25': forecast_p25,
            'forecast_p75': forecast_p75,
        }

        self.forecast = forecast_result

    def plot_forecast(self, show_samples=False):

        observed_time_series = np.exp(self.masked_series.time_series)
        # One Step Predictions
        one_step_mean = self.one_step['one_step_mean']
        one_step_p10 = self.one_step['one_step_p10']
        one_step_p90 = self.one_step['one_step_p90']
        one_step_p05 = self.one_step['one_step_p05']
        one_step_p95 = self.one_step['one_step_p95']
        one_step_p25 = self.one_step['one_step_p25']
        one_step_p75 = self.one_step['one_step_p75']
        one_step_median = self.one_step['one_step_median']

        # Smooth one_step_median:
        one_step_median_smooth = pd.Series(one_step_median).rolling(7, center=True).mean().values


        # Forecast
        forecast_mean = self.forecast['forecast_mean']
        forecast_p10 = self.forecast['forecast_p10']
        forecast_p90 = self.forecast['forecast_p90']
        forecast_p05 = self.forecast['forecast_p05']
        forecast_p95 = self.forecast['forecast_p95']
        forecast_p25 = self.forecast['forecast_p25']
        forecast_p75 = self.forecast['forecast_p75']
        forecast_median = self.forecast['forecast_median']


        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        c1, c2 = colors[0], colors[1]

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        num_steps = len(observed_time_series)
        num_steps_forecast = forecast_mean.shape[-1]

        ax.fill_between(np.arange(num_steps),
                            one_step_p05,
                            one_step_p95,
                            color=c1, alpha=0.1,
                            label='Fit 90% CI')

        ax.fill_between(np.arange(num_steps),
                            one_step_p10,
                            one_step_p90,
                            color=c1, alpha=0.2,
                            label='Fit 80% CI')

        ax.fill_between(np.arange(num_steps),
                            one_step_p25,
                            one_step_p75,
                            color=c1, alpha=0.3,
                            label='Fit 50% CI')

        ax.plot(np.arange(num_steps),
                one_step_median,
                lw=1,
                color=c1, alpha=0.5,
                label='Fit Median')

        ax.plot(np.arange(num_steps),
                one_step_median_smooth,
                lw=2,
                color=c1,
                label='Fit Median (7d MA)')

        ax.plot(np.arange(num_steps),
                observed_time_series,
                marker='o',
                linestyle='None',
                color=c1,
                label='Observed')


        forecast_steps = np.arange(
                num_steps,
                num_steps + num_steps_forecast)
        if show_samples:
            forecast_samples = self.forecast['forecast_samples']
            ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)

        ax.plot(forecast_steps, forecast_mean, lw=2, ls='--', color=c2,
                    label='Mean')
        #ax.plot(forecast_steps, forecast_median, lw=2, ls='-', color=c2, label='median')
        ax.fill_between(forecast_steps,
                            forecast_p05,
                            forecast_p95,
                            color=c2, alpha=0.1,
                            label='90% CI')
        ax.fill_between(forecast_steps,
                            forecast_p10,
                            forecast_p90,
                            color=c2, alpha=0.2,
                            label='80% CI')
        ax.fill_between(forecast_steps,
                            forecast_p25,
                            forecast_p75,
                            color=c2, alpha=0.3,
                            label='50% CI')

        ax.set_xlim([0, num_steps+num_steps_forecast])
        ymax = np.max(observed_time_series[np.isfinite(observed_time_series)]) * 1.1
        ax.set_ylim([0, ymax])

        # Generate date labels for x-ticks:
        start_date = self.df_merged['date'].min()
        end_date = self.df_merged['date'].max() + datetime.timedelta(days=num_steps_forecast)
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monthly start
        date_range_labels = date_range.strftime("%b %Y")
        xpos = (date_range - start_date).days

        ax.set_xticks(xpos)
        ax.set_xticklabels(date_range_labels)

        ax.set_xlabel("Date")
        ax.set_ylabel("Viral Copies per Person (SARS-CoV-2)")
        plt.suptitle("{}: COVID-19 Wastewater Trend".format(self.location), fontsize=16)
        plt.title(
            "Forecast from {} to {} using Bayesian Structural Time Series Model".format(
                self.df_merged['date'].max().strftime("%b %d"),
                end_date.strftime("%b %d")),
            fontsize=12
        )
        ax.legend(loc="upper left")

        caption = " Data Source: Alberta Health, Alberta Precision Laboratories, and Centre for Health Informatics"
        fig.text(0.95, 0.01, caption, ha='right', va='bottom',fontsize=8, color='gray')

        # Save plot to disk:
        plt.savefig('output/forecast_{}.png'.format(self.location))


    def export_csv(self):
        # Combine forecast with observed data:
        # - Create a dataframe with the forecast dates
        # - Combine with observed data
        # - Plot
        # - Export CSV
        forecast = self.forecast
        df_merged = self.df_merged

        # Create a dataframe with the forecast dates:
        df_forecast = pd.DataFrame(pd.date_range(start=df_merged['date'].max() + datetime.timedelta(days=1),
                                                 periods=len(forecast['forecast_mean']),
                                                 freq='D'),
                                   columns=['date'])
        df_forecast['value'] = forecast['forecast_mean']
        df_forecast['value_p10'] = forecast['forecast_p10']
        df_forecast['value_p90'] = forecast['forecast_p90']
        df_forecast['value_p05'] = forecast['forecast_p05']
        df_forecast['value_p95'] = forecast['forecast_p95']
        df_forecast['value_p25'] = forecast['forecast_p25']
        df_forecast['value_p75'] = forecast['forecast_p75']
        df_forecast['value_median'] = forecast['forecast_median']

        # Combine with observed data:
        df_combined = pd.concat([df_merged, df_forecast])

        # Export CSV:
        df_combined.to_csv('output/forecast_{}.csv'.format(self.location), index=False)
