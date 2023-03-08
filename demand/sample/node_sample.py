from typing import Optional
import pandas as pd


class NodeSample:
    """
    A class representing a node in a system with associated samples.

    Attributes:
        node_id (str): The ID of the node.
        samples (list): A list of samples associated with the node.
        sample_df (pandas.DataFrame): A Pandas DataFrame of the samples.

    Methods:
        __init__(node_id: str, samples: Optional[list] = None):
            Initializes a new instance of the NodeSample class.
        add_samples(new_samples):
            Adds new samples to the existing samples.
        get_df():
            Returns a Pandas DataFrame of the samples.
        get_sample_sum_between(start_date, end_date, mode='left'):
            Returns the sum of samples between two dates.
        get_df_with_range(start_date, end_date, mode='left'):
            Returns a Pandas DataFrame of the samples within a specified time range.
        get_agg_df(agg_para='D', start_date: Optional[str] = None, end_date: Optional[str] = None):
            Returns a Pandas DataFrame of the samples aggregated over a specified time range.
        get_sample_mean(agg_para='D', start_date: Optional[str] = None, end_date: Optional[str] = None):
            Returns the mean of the aggregated samples over a specified time range.
        get_sample_std(agg_para='D', start_date: Optional[str] = None, end_date: Optional[str] = None):
            Returns the standard deviation of the aggregated samples over a specified time range.
        get_sample_quantile(quantile=0.5, agg_para='D', start_date: Optional[str] = None, end_date: Optional[str] = None):
            Returns the specified quantile of the aggregated samples over a specified time range.
    """

    def __init__(self, node_id: str, samples: Optional[list] = None):
        """
        Initializes a new instance of the NodeSample class.

        Args:
            node_id (str): The ID of the node.
            samples (Optional[list]): A list of samples associated with the node.
        """
        self.node_id = node_id
        if samples is None:
            self.samples = []
        else:
            self.samples = samples
        self.sample_df = None

    def add_samples(self, new_samples):
        """
        Adds new samples to the existing samples.

        Args:
            new_samples: The new samples to add.
        """
        self.samples.extend(new_samples)

    def get_df(self):
        """
        Returns a Pandas DataFrame of the samples.

        Returns:
            pandas.DataFrame: A Pandas DataFrame of the samples.
        """
        sample_df = pd.DataFrame(self.samples, columns=['date', 'qty', 'from'])
        sample_df['date'] = pd.to_datetime(sample_df['date'])
        sample_df = sample_df.set_index('date')
        sample_df = sample_df.sort_index()
        self.sample_df = sample_df
        return sample_df

    def get_sample_sum_between(self, start_date, end_date, mode='left'):
        """
        Returns the sum of samples between two dates.

        Args:
            start_date: The start date.
            end_date: The end date.
            mode (str): The inclusive mode of the time range ('left' for left-inclusive, 'right' for right-inclusive).

        Returns:
            float: The sum of samples between the two dates.
        """
        df = self.get_df_with_range(start_date, end_date, mode)
        sample_sum = df['qty'].sum()
        return sample_sum

    def get_df_with_range(self, start_date, end_date, mode='left'):
        if self.sample_df is None:
            self.get_df()
        if mode == 'left':
            # (left, right]
            start_date = start_date + pd.Timedelta(seconds=1)
        df = self.sample_df.loc[start_date: end_date]
        return df

    def get_agg_df(self, agg_para='D', start_date: Optional[str] = None, end_date: Optional[str] = None):
        if self.sample_df is None:
            self.get_df()
        if start_date is None and end_date is None:
            agg_df = self.sample_df.resample(agg_para).sum()
        else:
            if start_date is None:
                start_date = self.sample_df.index.min().strftime(format='%Y-%m-%d')
            if end_date is None:
                end_date = self.sample_df.index.max().strftime(format='%Y-%m-%d')

            sample_df = self.get_df_with_range(start_date, end_date)
            t_index = pd.date_range(start=start_date, end=end_date, freq=agg_para)
            agg_df = sample_df.resample(agg_para).sum().reindex(t_index).fillna(0)

        return agg_df

    def get_sample_mean(self, agg_para='D', start_date: Optional[str] = None, end_date: Optional[str] = None):
        agg_sample_df = self.get_agg_df(agg_para, start_date, end_date)
        sample_mean = agg_sample_df.mean().values[0]
        return sample_mean

    def get_sample_std(self, agg_para='D', start_date: Optional[str] = None, end_date: Optional[str] = None):
        agg_sample_df = self.get_agg_df(agg_para, start_date, end_date)
        sample_std = agg_sample_df.std().values[0]
        return sample_std

    def get_sample_quantile(self, quantile=0.5, agg_para='D', start_date: Optional[str] = None,
                            end_date: Optional[str] = None):
        agg_sample_df = self.get_agg_df(agg_para, start_date, end_date)
        sample_quantile = agg_sample_df.quantile(quantile).values[0]
        return sample_quantile


