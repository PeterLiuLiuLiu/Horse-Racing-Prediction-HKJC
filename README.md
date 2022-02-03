# Horse-Racing-Prediction-HKJC

This project has been a working implementation of Computer Based Horse Race Handicapping and Wagering Systrems: A Report" (William Benter, 1994) url: https://www.gwern.net/docs/statistics/decision/1994-benter.pdf

It starts by parsing the HKJC website and weather of that particular race data at ST and HV course with "Selenium", read the HTML source with regex and tabluate with pandas and numpy. Finally it implements the probabilistic calculations by Benter by:

1. using libraries such as Sklearn abd tensorflow
2. Working on the probabilistic mathematics from scratch
   and try to determine the if the result fits the tables stated in the paper.

**_Update from 2022_**
It is believed that HKJC has removed access for horse racing data from the public and one can no longer access the years of data from its website API.

Data obtained by web_scrapping previously scrapped from year 2000 to 2020, 20 years worth of data from multiply branches of HKJC races in Happy Valley and Sha Tin race track.
