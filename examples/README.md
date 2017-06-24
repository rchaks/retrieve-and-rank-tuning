# Sample notebooks/scripts 
Walks through steps for creating, evaluating & experimenting with 
 RnR (Retrieve and Rank) as well as Discovery instances.

So far, I've got:
* 1.0 - Create RnR Cluster & Train Ranker
* 2.0 - Evaluate RnR Performance
* 3.0 - Create Discovery Collection & Evaluate Performance

Still to come:
* Evaluate Discovery Performance with Ranker
* Add custom feature to the Ranker

## The Data 
For the samples, we make use of the InsuranceLibV2 data: 
https://github.com/shuzi/insuranceQA.  At a high level, 
the InsuranceLibV2 is a _Question Answering_ data set provided for 
benchmarking and research, it consists of question and answers 
collected from the [Insurance Library](http://www.insurancelibrary.com/).

 - There are **27,413 possible answers** in this extract and each 
 looks something like: 

> Coverage follows the car. Example 1: if you were given a car (loaned) 
and the car has no insurance, you can buy insurance on the car and your
 insurance will be primary. Another option, someone helped you to buy 
 a car. For example your credit score isn't good enough to finance, so 
 a friend of yours signed under your loan as a primary payor. You can 
 get insurance under your name and even list your friend on the policy 
 as a loss payee. In this case, we always suggest you get a loan gap 
 coverage: the difference between the car's actual cash value and the 
 amount still owned on it. Example 2: the car you are loaned has 
 insurance. You can buy a policy under your name, list the car on 
 that policy and in case of the accident, your policy will become 
 a secondary or excess. Once the limits of the primary car insurance 
 are exhausted, your coverage would kick in and hopefully pay for the 
 rest. I specifically used the word hopefully, because each accident 
 is unique and it's hard to interpret the coverage without the actual
  claim scenario. And even with a given claim scenario, sometimes 
  there are 2 possible outcomes of a claim.

 - In addition, a **16,899 questions** have been labelled with the 
 answer ids that are relevant to the question. We use a subset of 
 those questions (specifically the **2,000 question dev subset** 
 for speedy experiments) as input to train a ranker. The questions 
 look like this:
 
> Does auto insurance go down when you turn 21?

More info at: https://github.ibm.com/rchakravarti/rnr-debugging-scripts/tree/master/resources/insurance_lib_v2 