# Portfolio Optimization via Machine Learning Models

This repository contains the code, the results and the report of a project called Portfolio Optimization via Machine Learning Models, developed by three master students in Artificial Intelligence at University of Bologna.

## Abstract

In this study, we tackle two main challenges in quantitative finance: selecting an optimal stock portfolio and improving returns annually. To achieve this, we leverage various clustering techniques to identify promising shares for the upcoming month across different markets. We then optimize portfolio weights based on the Sharpe ratio. Finally, we tune different models to guide investment decisions for the following month. Our results reveal superior performance compared to the FTSEMIB index in the Italian market, while achieving outcomes comparable to the S&P 500. This difference may be attributed to the higher liquidity in the American one, renowned for its resilience during crashes and rapid recovery mechanisms.

<figure>
  <img src="Images/System_description.png" alt="System_description">
  <figcaption>Our proposed pipeline.</figcaption>
</figure>


## Results
Results are stored in the `Results` folder:
- `Results/Image/` contains the plots for the obtained compound returns
- `Results/Returns/` contains the calculated returns for each tested method
- `Results/American_market.xlsx` and `Results/Italian_market.xlsx` store the evaluation metrics on the returns.

Further analysis of the results can be found in the project report (`report.pdf`).<br>

#### Example: DBSCAN in the validation dataset (years 2016 - 2024)

<figure>
  <img src="Results\Images\IT_dbscan_val.png" alt="System_description">
  <figcaption>Compound returns for our proposed models in the Italian market between 2016-today</figcaption>
</figure>

<center>

| DBSCAN - Italy        | ROI   | σ     | Sharpe Ratio |
|--------------|-------|-------|--------------|
| Base         | 4.7%  | 0.015 | 0.009        |
| Persistency  | 57.9% | 0.011 | 0.025        |
| Regression   | 12.2% | 0.013 | 0.011        |
| Arima        | 156%  | 0.01  | 0.044        |
| Mean         | 57.8% | 0.013 | 0.022        |
| FTSE MIB     | 17.9% | 0.014 | 0.013        |
| Random       | 2.7%  | 0.008 | 0.004        |

</center>


<figure>
  <img src="Results\Images\US_dbscan_val.png" alt="System_description">
  <figcaption>Compound returns for our proposed models in the American market between 2016-today</figcaption>
</figure>

<center>

| DBSCAN -USA          | ROI   | σ     | Sharpe Ratio |
|--------------|-------|-------|--------------|
| Base         | 32.8% | 0.015 | 0.017        |
| Persistency  | 42.0% | 0.011 | 0.021        |
| Regression   | 116%  | 0.011 | 0.041        |
| Arima        | 116%  | 0.010 | 0.042        |
| Mean         | 76.7% | 0.012 | 0.030        |
| S&P 500      | 141%  | 0.012 | 0.043        |
| Random       | 17.9% | 0.008 | 0.015        |

</center>


## Run using conda
An easy way to reproduce the results is to create your own working environment using conda, typing the following commands.

1. Create and enter a new conda environment<br>
``` conda create --name PortfolioOptimization python=3.11 ``` <br>
``` conda activate PortfolioOptimization ``` 

2. Install the required dependencies<br>
``` pip install -r requirements.txt```

3. Open Jupyter Notebook, connect to the PortfolioOptimization kernel and run.

<!-- ## Key Features

- Implementation of clustering techniques (DBSCAN, K-means) for stock selection.
- Portfolio optimization based on the Sharpe ratio
- Regression and ARIMA models for trend prediction and decision-making.
- Evaluation and comparison of results in the Italian and American markets. -->




<!-- ## Folder Structure

- `Dataset`: Contains the utility functions and the metrics used to build our costum dataset
- `Optimization`: Contains the clustering techniques used and the portoflio optimization done over the sharpe ratio.
- `Evaluation`: Contains the evaluation metrics and the plots.
- `Report`: Project paper. 
- `Italian_market`: Notebooks specific to the Italian market.
- `American_market`: Notebooks specific to the American market.
- `Images`: Images used in documentation and visualizations.
 -->

## Contributors

- [Francesco Pivi](https://github.com/your-username)
- [Elisa Castanari](https://github.com/teammate1)
- [Matteo Fusconi](https://github.com/teammate2)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to Luca Giuliani for guidance and support throughout the project.
- Thanks to University of Bologna for providing resources and facilities for conducting research.


