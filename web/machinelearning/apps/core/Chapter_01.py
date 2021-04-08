from .utlities import Algo
import pandas as pd


class Algo1(Algo):
    def __init__(self):
        super().__init__(chapter_id="CH01_fundamentals", to_data_path="lifesat", target_field=None)
        print('-'*50)
        print('in constructor')
        print('----   oecd  ------')
        oecd_bli = pd.read_csv(self.TO_DATA_PATH + "/oecd_bli_2015.csv", thousands=',')
        print(oecd_bli)
        print('-'*50)
        gdp_per_capita = pd.read_csv(self.TO_DATA_PATH + "/gdp_per_capita.csv", thousands=',', delimiter='\t',
                                     encoding='latin1', na_values="n/a")
        print('='*50)
        print('gdp_per_capita')
        print(gdp_per_capita)
        print('-'*50)
        self.full_country_stats, self.sample_data, self.missing_data = self.prepare_country_stats(oecd_bli, gdp_per_capita)

    def prepare_country_stats(self, oecd_bli, gdp_per_capita):
        print('in prepare_country_stats')
        print('-'*50)
        print('- Before filtering -')
        print(oecd_bli)
        oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
        print('- After filtering')
        print('-'*20)
        print('oecd_bli')
        print(oecd_bli)
        print(oecd_bli.shape)
        print('-'*50)
        oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
        print('After Pivot')
        print('-'*10)
        print(oecd_bli)
        print('-'*50)
        print('gdp_per_capita')
        print('-'*20)
        print(gdp_per_capita)
        gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
        print('-- after rename --')
        print(gdp_per_capita)
        print('-'*5)
        gdp_per_capita.set_index("Country", inplace=True)
        print('-- after set index --')
        print(gdp_per_capita)
        full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                      left_index=True, right_index=True)
        print('-- after merge --')
        print('full_country_stats')
        print(full_country_stats)

        full_country_stats.sort_values(by="GDP per capita", inplace=True)
        print('-- after sort value by GDP per capita --')
        print(full_country_stats)

        remove_indices = [0, 1, 6, 8, 33, 34, 35]
        keep_indices = list(set(range(36)) - set(remove_indices))

        print('keep_indices')
        print(keep_indices)
        print('keep_indices')

        print('-'*50)
        print("full_country_stats[['GDP per capita', 'Life satisfaction']]")
        print(full_country_stats[["GDP per capita", 'Life satisfaction']])
        print(full_country_stats[["GDP per capita", 'Life satisfaction']].shape)
        print('-'*50)
        print('-'*50)
        print("full_country_stats[['GDP per capita', 'Life satisfaction']].iloc[keep_indices]")
        print(full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices])
        print(full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices].shape)
        print('-'*50)
        print("full_country_stats[['GDP per capita', 'Life satisfaction']].iloc[remove_indices]")
        print(full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indices])
        print(full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indices].shape)
        print('-'*50)

        return full_country_stats[["GDP per capita", 'Life satisfaction']], \
               full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices], \
               full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indices]

