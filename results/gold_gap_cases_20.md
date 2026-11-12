# Gold-Wrong Case Dump — 20 claims (seed=42)

Source: `results/gold_baseline_official_dev_with_metadata_cache.json` filtered to `pred != true_label` (or unparseable). Draw uniform at random from the 111 gold-wrong claims.

For each case, classify into ONE of:
- **(a) Reasoning** — evidence clearly supports one label; LLM chose another.
- **(b) Annotation** — the true label is a judgment call; either label is defensible.
- **(c) Prompt** — the plain prompt is too terse for this claim type (e.g. Conflicting needs a per-side reasoning protocol).
- **(d) Format** — the QA-pair format scatters or hides the annotator's key signal (a raw source page would have made the answer obvious).

---

## Case 1 — claim_5 (dev.json idx 5)

- **true**: Refuted   →   **gold-pred**: NEE
- **claim**: The United States of America and its Western allies have been using their media outlets to publish articles based on fabricated information under allegations of non-compliance with the Chemical Weapons Convention.
- **speaker / source**: 'Syrian Arab News Agency (SANA)' / 'Syrian state media outlet'
- **annotator justification**: Refuted by answer/source.

**Gold QA pairs:**
- Q1: Has Syria complied with the Chemical Weapons Convention?
  - A: No

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 2 — claim_8 (dev.json idx 8)

- **true**: Refuted   →   **gold-pred**: Supported
- **claim**: You’re watching the cheaters and all those people that send in the phony ballots. … They want to have the count weeks after November 3.
- **speaker / source**: 'Donald Trump' / ''
- **annotator justification**: The mail in ballots have to be counted no later than 7 days after election day. so that it would be weeks later is refuted.

**Gold QA pairs:**
- Q1: When are mail-in ballots counted?
  - A: Vote-by-mail ballots that are mailed must be postmarked on or before Election Day and received by your county elections office no later than 7 days after Election Day.
- Q2: are all non-mail ballots counted by election day?
  - A: No

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 3 — claim_31 (dev.json idx 31)

- **true**: Supported   →   **gold-pred**: NEE
- **claim**: Amy Coney Barrett was confirmed as US Supreme Court Justice on October 26, 2020
- **speaker / source**: 'Meena Harris @meenaharris' / 'Twitter'
- **annotator justification**: Although the date is not confirmed in the QA pair, it is stated in the source for the Answer provided.

**Gold QA pairs:**
- Q1: Is Amy Coney Barrett confirmed as supreme Court justice ?
  - A: Yes

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 4 — claim_41 (dev.json idx 41)

- **true**: Refuted   →   **gold-pred**: NEE
- **claim**: Wearing face masks can cause infections from bacteria such as staphylococcus.
- **speaker / source**: 'Jane Taylor' / 'Facebook'
- **annotator justification**: Prolonged mask use can cause side effects such as acne, difficulty breathing, headaches, impaired cognition, rashes, and skin breakdown. It also interferes with vision, communication, and thermal balance. There isn't evidence it can cause staphylococcus. Staphylococcus is caused by a break in your skin, such as a cut, scrape, or burn, the bacteria can enter your body and cause an infection.

**Gold QA pairs:**
- Q1: Does wearing face mask  cause bacterial infection (staphylococcus ) on your face ?
  - A: Wearing masks for a long time causes a lot of physiological and psychological loads and can reduce work efficiency. The activity cannot be performed as much or as efficiently while wearing the mask as when the masks are not worn. In addition, when you can continue an activity, it decreases when using the mask. Prolonged use of N95 and surgical masks can cause physical side effects such as acne, difficulty breathing, headaches, impaired cognition, rashes, and skin breakdown. It also interferes wi…
- Q2: What  cause ‘staph’ infections?
  - A: You probably come into contact with staph bacteria on a daily basis, but it doesn’t always cause an infection. If you have a break in your skin, though, such as a cut, scrape, or burn, the bacteria can enter your body and cause an infection.

There are several things that can cause a a break in the delicate skin inside your nose, including:

   excessive nose blowing
    picking your nose
    plucking or tweezing your nose hairs
    having a nose piercing

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 5 — claim_43 (dev.json idx 43)

- **true**: Conflicting   →   **gold-pred**: Refuted
- **claim**: Donald Trump said that $15 an hour is too much for essential workers
- **speaker / source**: 'Joe Biden' / ''
- **annotator justification**: Even though Trump said similar to this, it was taking out of context. The claim was cherrypicking.

**Gold QA pairs:**
- Q1: Did Donald Trump say that $15 an hour is too much for essential workers?
  - A: Trump Says He’d ‘Consider’ $15 Per Hour Federal Minimum Wage, But Prefers To Leave It To States

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 6 — claim_56 (dev.json idx 56)

- **true**: Refuted   →   **gold-pred**: NEE
- **claim**: Previous flu vaccines may have made President Donald Trump susceptible to COVID-19
- **speaker / source**: 'Miriam Fauzia' / 'USA TODAY'
- **annotator justification**: Studys show the flu vaccine does not affect how susceptible some one is to covid-19 so based on this it would mean Donald Trump was not more susceptible.

**Gold QA pairs:**
- Q1: What is the difference between Flu and Covid-19?
  - A: COVID-19 and influenza (flu) are both infectious respiratory illnesses. Though the symptoms of these two diseases seem similar, they are caused by different viruses. While many people often compare COVID-19 to the flu, there are many differences between them.
- Q2: Have any studies been done to show if having the Flu vaccine makes you more susceptible  to getting Covid-19?
  - A: Yes

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 7 — claim_90 (dev.json idx 90)

- **true**: Refuted   →   **gold-pred**: NEE
- **claim**: BJP MP Kirron Kher Said "Rape Is A Part Of Indian Culture And It Can't Be Stopped"
- **speaker / source**: '' / 'Facebook'
- **annotator justification**: The MP has said comments about rape but not the phrase that it can not be stopped. The claim was also on a Facebook page that spreads fake news.

**Gold QA pairs:**
- Q1: Where was this claim published?
  - A: Viral In India
- Q2: What is Viral In India?
  - A: A Facebook page known for spreading fake news.
- Q3: Has BKP MP Kirron Kher spoken out about rape?
  - A: Yes
  - A: Yes

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 8 — claim_95 (dev.json idx 95)

- **true**: Refuted   →   **gold-pred**: NEE
- **claim**: Controversial abortion methods are allowed under New Zealand's Abortion Legislation Act (2020).
- **speaker / source**: 'Right To Life UK' / 'Facebook'
- **annotator justification**: The answers and credible sources show that they do not have controversial abortion methods so the claim is refuted.

**Gold QA pairs:**
- Q1: What is the New Zealand's Abortion Legislation Act (2020)?
  - A: The Abortion Legislation Act (2020), which came into effect March 2020, decriminalised abortion in New Zealand.
- Q2: Was abortion legal in New Zealand before the New Zealand's Abortion Legislation Act (2020)?
  - A: Abortions were regulated by the New Zealand Crime act from 1961. Before the Abortion Legislation Act 2020 came into effect the Crimes Act 1961 only explicitly allowed for abortions on disability grounds before 20 weeks and after 20 weeks were only permitted if it was “necessary to save the life of the woman or girl or to prevent serious permanent injury to her physical or mental health”.
- Q3: What are the changes to the abortion laws?
  - A: 10 Provision of abortion services to women not more than 20 weeks pregnant

A qualified health practitioner may provide abortion services to a woman who is not more than 20 weeks pregnant.
11 Provision of abortion services to women more than 20 weeks pregnant

(1)

A qualified health practitioner may only provide abortion services to a woman who is more than 20 weeks pregnant if the health practitioner reasonably believes that the abortion is clinically appropriate in the circumstances.

(2)

In…
  - A: Abortions are now permitted for women who are less than 20 weeks pregnant to be carried out by a qualified health practitioner and over 20 weeks a health practitioner  must reasonably believe that the abortion is clinically appropriate in the circumstances.

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 9 — claim_96 (dev.json idx 96)

- **true**: Refuted   →   **gold-pred**: NEE
- **claim**: Commission on Human Rights (CHR) Chairman Chito Gascon called a “surprise military attack” by the Philippine Army,  conducted on members of the New People’s Army (NPA) a “violation of human rights.”
- **speaker / source**: 'Pinoy News Center' / 'Blog'
- **annotator justification**: The answers show that the page is a fake news site so the claim is refuted.

**Gold QA pairs:**
- Q1: Where was the claim made initially
  - A: It was published on pinoynewscenter.blogspot.com
- Q2: What kind of website is pinoynewscenter.blogpost.com
  - A: It is a website known for publishing false claims

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 10 — claim_99 (dev.json idx 99)

- **true**: Refuted   →   **gold-pred**: Supported
- **claim**: New Zealand's Abortion Legislation Act (2020) does not make it a legal requirement that babies born alive after an abortion are given medical support.
- **speaker / source**: 'Right To Life UK' / 'Facebook'
- **annotator justification**: The answer and source show that the claim is not correct so it is refuted.

**Gold QA pairs:**
- Q1: Are babies born alive after an abortion in NZ entitled to  medical support?
  - A: Yes
- Q2: Does NZ  Abortion Legislation Act (2020) makes it a legal requirement that babies born alive after an abortion are given medical support ?
  - A: False , as per the New Zealand's Abortion Legislation Act (2020),  proposed amendment was voted down by 80 to 37 votes, so will not be part of the Bill.  If it was passed, then The amendment would have required that if a baby was born alive following a failed abortion, a qualified health practitioner or any other health practitioner present at the time the child is born, would have a duty to provide the child with appropriate medical care and treatment.

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 11 — claim_106 (dev.json idx 106)

- **true**: Refuted   →   **gold-pred**: NEE
- **claim**: Amy Coney Barrett, associate justice of the Supreme Court of the United States,  said during her Supreme Court confirmation hearing that she was not using notes because she has seven kids and is used to stupid questions.
- **speaker / source**: 'Becket Adams, a commentary writer for the Washington Examiner.' / 'Twitter'
- **annotator justification**: Claim Not supported by QA pairs.

**Gold QA pairs:**
- Q1: did Amy Coney Barrett use notes for the supreme court confirmation hearing
  - A: Sen. John Cornyn, R-Texas, said it was "impressive" that Supreme Court nominee Amy Coney Barrett was referring to zero notes at her confirmation hearing.
- Q2: what was Amy Coney Barrett's answer when asked by Senator Cornyn of the content of her notepad
  - A: She said "The letterhead says United States Senate,"

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 12 — claim_238 (dev.json idx 238)

- **true**: Refuted   →   **gold-pred**: Conflicting
- **claim**: The number of police decline when National Party leader Judith Collins’ was police minister.
- **speaker / source**: 'Jacinda Ardern' / 'Newshub Leaders Debate'
- **annotator justification**: Q&A 1 shows that when the person named in the claim (Ms Collins ) first became police minister in November 2008 there were 8,211 police officers. 
Q&A 2 then  shows that numbers then rose to 8,643 in June 2009 and 8,707 in June 2010, before dropping to 8,638 in June 2011
Q&A 3 shows that when she returned to the role that There were 8,923 police officers in June 2015 and this dropped to 8,899 in June 2016.when she left the role.
so whilst some drops were shown it was overall small, and there was a general increase over her total time so refuting the claim really.

**Gold QA pairs:**
- Q1: How many police officers were there in June 2008 when Ms collins became police minister?
  - A: According to NZ Police, in June 2008, there were 8,211 police officers. Ms Collins first became police minister in November 2008.
- Q2: How  many police officers were in 2011 when Ms Collin's portoflio's changed?
  - A: The NZ Police figures show officer numbers then rose to 8,643 in June 2009 and 8,707 in June 2010, before dropping to 8,638 in June 2011.
- Q3: How many police officers were there in 2015 when MS collins returns to the role of police minister?
  - A: There were 8,923 police officers in June 2015 and this dropped to 8,899 in June 2016. Ms Collins portfolios changed again in December 2016.
- Q4: How does this compare as as a ration of the resident population?
  - A: AAP FactCheck calculated police numbers as a ratio of the resident population for the years in which Judith Collins was police minister by dividing the population figures by the constabulary numbers for the corresponding year.

This revealed there was one police officer for every 519 NZ residents in June 2008, one for every 498 residents in June 2009, 1/500 in 2010, 1/507 in 2011, 1/514 in 2015 and 1/526 in 2016
- Q5: How many police officers were there in June 2008 when Ms Collins became police minister?
  - A: According to NZ Police, in June 2008, there were 8,211 police officers. Ms Collins first became police minister in November 2008.
- Q6: How  many police officers were in 2011 when Ms Collin's portfolio's changed?
  - A: The NZ Police figures show officer numbers then rose to 8,643 in June 2009 and 8,707 in June 2010, before dropping to 8,638 in June 2011.
- Q7: How many police officers were there in 2015 when Ms Collins returns to the role of police minister?
  - A: There were 8,923 police officers in June 2015 and this dropped to 8,899 in June 2016. Ms Collins portfolios changed again in December 2016.
- Q8: How does this compare as a ratio of the resident population?
  - A: AAP FactCheck calculated police numbers as a ratio of the resident population for the years in which Judith Collins was police minister by dividing the population figures by the constabulary numbers for the corresponding year.

This revealed there was one police officer for every 519 NZ residents in June 2008, one for every 498 residents in June 2009, 1/500 in 2010, 1/507 in 2011, 1/514 in 2015 and 1/526 in 2016

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 13 — claim_284 (dev.json idx 284)

- **true**: Refuted   →   **gold-pred**: NEE
- **claim**: Former US vice-president Joe Biden is showing pretty severe signs of dementia.
- **speaker / source**: 'thepeachfuzz' / 'Instagram'
- **annotator justification**: I feel that without access to Joe Biden's medical records and a statement from him or his doctor, there is no way to know as the diagnosis process is lengthy and the symptoms can vary and be indicative of other illnesses / conditions, such as his stutter which he has had for his whole life.

**Gold QA pairs:**
- Q1: How is dementia diagnosed?
  - A: Medical history and physical exam. 
Neurological evaluations. 
Brain scans.
Cognitive and neuropsychological tests. 
Laboratory tests. 
Presymptomatic tests. Genetic testing. 
Psychiatric evaluation.
- Q2: How long has Joe Biden had a stutter?
  - A: Former Vice President Joe Biden opened up about his lifelong struggle with stuttering on Wednesday, offering a rare lengthy and personal reflection on how it has affected him, even to this day.
  - A: Maybe you’ve heard Biden talk about his boyhood stutter.
- Q3: Has a medical doctor diagnosed Joe Biden with dementia?
  - A: US Congress member and  urologist, Dr Greg Murphy tweeted: Joe you are suffering from dementia. I am sorry but it is true.
  - A: According to another fact-check source:
Greg Murphy... has said repeatedly ... that Biden... has dementia.
Murphy is a medical doctor — a urologist.
  - A: According to another fact-check source:
He (Dr Greg Murphy )cited Biden’s speech patterns

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 14 — claim_324 (dev.json idx 324)

- **true**: Conflicting   →   **gold-pred**: Refuted
- **claim**: US Miami-Dade County Mayor Carlos Giminez kept his taxpayer-funded Mercedes-Benz after slashing $400 million in government jobs and spending.
- **speaker / source**: 'Debbie Mucarsel-Powell' / 'Media Cntre'
- **annotator justification**: Q&A 3 state that he never had the car allowance when mayor, but when commissioner.

**Gold QA pairs:**
- Q1: When did US Miami-Dade County Mayor Carlos Giminez
  - A: J.C. Planas, an attorney for the Gimenez campaign and a former state legislator, told PolitiFact in a previous fact-check on June 20, 2011 that Gimenez was entitled to the same car allowances as all the commissioners. For the last year and a half, Gimenez drove a Mercedes-Benz ML and prior to that a Mercedes-Benz E320...Their salary is their benefits -- one of which is their car allowance.
The fact-checking agency also stated that:  Car allowances -- or the use of city-owned cars -- is one piece…
- Q2: What has been published or verified in other media reports or documentation?
  - A: No answer could be found.
- Q3: When did Gimenez drive a Mercedes
  - A: WHen he was a county commissioner, he has car allowance. He approved the cut of 400$ as county comssision official, not as mayor
- Q4: Does Gimenex have a car allowance currently?
  - A: No

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 15 — claim_330 (dev.json idx 330)

- **true**: Supported   →   **gold-pred**: NEE
- **claim**: Ventura County election officials are reducing voting locations in order to limit in-person voter turnout.
- **speaker / source**: 'Bill Gentry' / 'Facebook'
- **annotator justification**: Facing the uncertainty of an election conducted during the COVID-19 pandemic, state officials are encouraging vote-by-mail and sending every California voter a ballot

**Gold QA pairs:**
- Q1: Is California reducing its amount of voting stations?
  - A: Reductions in voting stations are across the board some examples are Sacramento County opened 84 vote centers for the March primary, but has only secured 39 sites for the November election. In Santa Barbara, officials are reducing the number of in-person voting sites from 86 to 25.
- Q2: Why are Voting station numbers being reduced?
  - A: Facing the uncertainty of an election conducted during the COVID-19 pandemic, state officials are encouraging vote-by-mail and sending every California voter a ballot

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 16 — claim_350 (dev.json idx 350)

- **true**: Conflicting   →   **gold-pred**: Supported
- **claim**: The rate of abortions under Democratic presidential administrations has actually fallen more precipitously.
- **speaker / source**: '' / 'Facebook'
- **annotator justification**: While the QA pairs do support lower levels of abortion during Democratic administrations , they also state there were fewer pregnancies.

**Gold QA pairs:**
- Q1: How is data on abortion collected?
  - A: Since 1969, the CDC has attempted to collected nationwide data to document the number of legal abortions, the abortion ratio (abortions versus live births), and the abortion rate (number of abortions per 1,000 women ages 15-44). 

But the CDC data isn’t complete because not all abortions are counted.  

The latest CDC abortion report, from 2016, used data from health agencies of 48 reporting areas (New York City and 47 states). But it did not include California, the District of Columbia, Marylan…
- Q2: Has the abortion rate decreased in recent years?
  - A: In 2017, the organization reported that the U.S. abortion rate had decreased to record low levels, dropping to 13.5 abortions per 1,000 women aged 15–44, the lowest rate recorded since abortion was legalized in 1973.
- Q3: Was the decline greater in a democratic president?
  - A: Yes
- Q4: What factors influence abortion incidence?
  - A: "Between 2011 and 2017, the abortion decline appears to be driven by fewer people becoming pregnant in the first place, as both abortions and births declined. If abortion restrictions were the main driver across the board, we’d expect the birth rate to increase," Nash said. "There are a lot of potential factors at play, including increased access to affordable contraception (due to the Affordable Care Act among other policies that make it easier to afford contraception), new methods of contracep…

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 17 — claim_366 (dev.json idx 366)

- **true**: Conflicting   →   **gold-pred**: Supported
- **claim**: Biden called law enforcement the enemy.
- **speaker / source**: 'Donald Trump' / 'Rev'
- **annotator justification**: The claim Biden made was taken out of context, he was saying that IF law enforcement use equipment that is commonly seen in warzones, there will be a stronger divide and the police become the enemy

**Gold QA pairs:**
- Q1: in What context did Biden say the law enforcement "the enemy"?
  - A: Biden said that local law enforcement using “an up-armored Humvee” in a neighborhood looks “like the military invading,” adding, “They become the enemy.”
- Q2: What does Biden say about the police?
  - A: Biden campaign spokesperson Rosemary Boeglin told us in a statement: “For years, Biden has opposed providing police with equipment that only belongs in war zones — and was remarking that doing so can create and deepen divides between communities and the officers who serve them. That’s why Biden has long advocated for community policing strategies that strengthen relationships and foster trust.”
- Q3: Was Joe Biden's comment on law enforcement taken out of context?
  - A: Yes

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 18 — claim_383 (dev.json idx 383)

- **true**: Refuted   →   **gold-pred**: NEE
- **claim**: Asha Ranaut tweeted ‘you havent crushed my daughter’s office but Bal Thackeray’s soul.
- **speaker / source**: 'Times Now' / 'Times Now'
- **annotator justification**: From the 2nd QA it is evident that the claim actually false. Therefore the label is refuted

**Gold QA pairs:**
- Q1: Which account made the tweet ?
  - A: @RealAshaRanaut
- Q2: What is known about the twitter account @RealAshaRanaut?
  - A: The account was earlier called DrmohanRss1 and is a fake.

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 19 — claim_410 (dev.json idx 410)

- **true**: Refuted   →   **gold-pred**: NEE
- **claim**: Maharashtra Navnirman Sena (MNS) chief Raj Thackeray has come forward in support of Bollywood actress Kangana Ranaut and would welcome her back to Mumbai, India.
- **speaker / source**: 'Raj Thackeray' / 'Twitter'
- **annotator justification**: The twitter account is not the original twitter account of Raj Thsckery

**Gold QA pairs:**
- Q1: Where did the claim originate?
  - A: From the twitter account 
@thackerayoffic on Sep 4 2020.
- Q2: What is Raj Thackeray's Twitter account username?
  - A: @RajThackeray

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---

## Case 20 — claim_448 (dev.json idx 448)

- **true**: Conflicting   →   **gold-pred**: Supported
- **claim**: Kyle Rittenhouse failed to appear at his court hearing in Lake County, Illinois, on 28th August 2020 and was given an extra month to comply.
- **speaker / source**: 'Emilie Marie Blake' / 'Facebook'
- **annotator justification**: He did not appear at the hearing but that was because he was looking for an attorney so the claim is cherry picking.

**Gold QA pairs:**
- Q1: Did Kyle Rittenhouse appear for his first hearing on 28 August 2020?
  - A: No
- Q2: Was Rittehouse's initial hearing delayed by the judge?
  - A: Yes

**Classification**: _(a) / (b) / (c) / (d)_ — one-line reason:

---
