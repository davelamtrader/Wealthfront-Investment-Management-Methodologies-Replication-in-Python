# Wealthfront-Investment-Management-Methodologies-Replication-in-Python

This article combines the Wealthfront investment whitepaper and provides a detailed introduction to Wealthfront's asset allocation method, which is now open source. 
For specific details on the Wealthfront investment whitepaper, please refer to the link: https://research.wealthfront.com/whitepapers/investment-methodology/

***Introduction to Wealthfront***
Wealthfront is a well-known online asset management platform in the United States, with assets under management exceeding $50 billion. You can learn more about Wealthfront at https://www.wealthfront.com/

Using ETFs as the underlying assets, Wealthfront constructs different investment portfolios based on clients' varying risk preferences. 
It continuously monitors users' portfolio holdings, provides health scores, and adjusts holdings to optimal positions based on market conditions and changes in client risk preferences in real time.

***Investment Philosophy***
1. Value Investing (Long-term Investment): Enjoy capital appreciation brought by economic growth. Not everyone has time to actively trade, and short-term investments can be exhausting and unreliable.
2. Passive Investing: Numerous studies both domestically and internationally indicate that, in the long run, active investing may not necessarily outperform passive investing. Passive investing also makes it easier to diversify risk.
3. Asset Allocation: Don't put all your eggs in one basket. Proper asset allocation helps diversify away non-systemic risks.

Next, we will provide a detailed description of the complete investment process, including selecting asset classes, correlation matrices, 
constructing efficient frontiers, asset allocation methods, portfolio monitoring, and dynamic rebalancing. We will also incorporate specific examples 
based on the Chinese market situation to illustrate the above process.
