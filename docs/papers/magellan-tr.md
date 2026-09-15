# **Magellan: Toward Building Entity Matching Management Systems**

[Technical Report]

Pradap Konda<sup>1</sup> , Sanjib Das<sup>1</sup> , Paul Suganthan G.C.<sup>1</sup> , AnHai Doan<sup>1</sup> , Adel Ardalan<sup>1</sup> , Jeffrey R. Ballard<sup>1</sup> , Han Li<sup>1</sup> , Fatemah Panahi<sup>1</sup> , Haojun Zhang<sup>1</sup> , Jeff Naughton<sup>1</sup> , Shishir Prasad<sup>3</sup> , Ganesh Krishnan<sup>2</sup> , Rohit Deep<sup>2</sup> , Vijay Raghavendra<sup>2</sup>

1University of Wisconsin-Madison, 2@WalmartLabs, 3Instacart _∗_

## **ABSTRACT**

Entity matching (EM) has been a long-standing challenge in data management. Most current EM works focus only on developing matching algorithms. We argue that far more efforts should be devoted to building EM systems. We discuss the limitations of current EM systems, then present as a solution Magellan, a new kind of EM systems. Magellan is novel in four important aspects. (1) It provides how-to guides that tell users what to do in each EM scenario, step by step. (2) It provides tools to help users do these steps; the tools seek to cover the entire EM pipeline, not just matching and blocking as current EM systems do. (3) Tools are built on top of the data analysis and Big Data stacks in Python, allowing Magellan to borrow a rich set of capabilities in data cleaning, IE, visualization, learning, etc. (4) Magellan provides a powerful scripting environment to facilitate interactive experimentation and quick “patching” of the system. We describe research challenges raised by Magellan, then present extensive experiments with 44 students and users at several organizations that show the promise of the Magellan approach.

## **1. INTRODUCTION**

Entity matching (EM) identifies data instances that refer to the same real-world entity, such as (David Smith, UWMadison) and (D. M. Smith, UWM). This problem has been a long-standing challenge in data management [16, 22]. Most current EM works however has focused only on developing _matching algorithms_ [16, 22].

Going forward, we believe that building EM systems is truly critical for advancing the field. EM is engineering by nature. We cannot just keep developing matching algorithms in a vacuum. This is akin to continuing to develop

> _∗_ Work done while at WalmartLabs

join algorithms without having the rest of the RDBMSs. At some point we must build end-to-end systems to evaluate matching algorithms, to integrate research and development efforts, and to make practical impacts.

In this aspect, EM can take inspiration from RDBMSs and Big Data systems. Pioneering systems such as System R, Ingres, and Hadoop have really helped push these fields forward, by helping to evaluate research ideas, providing an architectural blueprint for the entire community to focus on, facilitating more advanced systems, and making widespread real-world impacts.

The question then is what kinds of EM systems we should build, and how? In this paper we begin by showing that current EM systems suffer from four limitations that prevent them from being used extensively in practice.

First, when performing EM users often must execute many steps, e.g., blocking, matching, exploring, cleaning, debugging, sampling, labeling, estimating accuracy, etc. Current systems however do not cover the entire EM pipeline, providing support for only a few steps (e.g., blocking, matching), while ignoring less well-known yet equally critical steps (e.g., debugging, sampling).

Second, EM steps often exploit many techniques, e.g., learning, mining, visualization, outlier detection, information extraction (IE), crowdsourcing, etc. Today however it is very difficult to exploit a wide range of such techniques. Incorporating all such techniques into a single EM system is extremely difficult. EM is often an iterative process. So the alternate solution of moving data repeatedly among an EM system, a data cleaning system, an IE system, etc. does not work either, as it is tedious and time consuming. A major problem here is that most current EM systems are standalone monoliths that are not designed from the scratch to “play well” with other systems.

Third, users often have to write code to “patch” the system, either to implement a lacking functionality (e.g., extracting product weights) or to glue together system components. Ideally such coding should be done using a script language in an interactive environment, to enable rapid prototyping and iteration. Most current EM systems however do not provide such facilities.

Finally, in many EM scenarios users often do not know what steps to take. Suppose a user wants to perform EM with at least 95% precision and 80% recall. Should he or she use a learning-based EM approach, a rule-based approach,

1

or both? If learning-based, then which technique to select among the many existing ones (e.g., decision tree, SVM, etc.)? How to debug the selected technique? What to do if after many tries the user still cannot reach 80% recall with a learning-based approach? Current EM systems provide no answers to such questions.

**The Magellan Solution:** To address these limitations, we describe Magellan, a new kind of EM systems currently being developed at UW-Madison, in collaboration with WalmartLabs. Magellan (named after Ferdinand Magellan, the first end-to-end explorer of the globe) is novel in several important aspects.

First, Magellan provides how-to guides that tell users what to do in each EM scenario, step by step. Second, Magellan provides tools that help users do these steps. These tools seek to cover the entire EM pipeline (e.g., debugging, sampling), not just the matching and blocking steps.

Third, the tools are being built on top of the Python data analysis and Big Data stacks. Specifically, we propose that users solve an EM scenario in two stages. In the development stage users find an accurate EM workflow using data samples. Then in the production stage users execute this workflow on the entirety of data. We observe that the development stage basically performs data analysis. So we develop tools for this stage on top of the well-known Python data analysis stack, which provide a rich set of tools such as pandas, scikit-learn, matplotlib, etc. Similarly, we develop tools for the production stage on top of the Python Big Data stack (e.g., Pydoop, mrjob, PySpark, etc.).

Thus, Magellan is well integrated with the Python data eco-system, allowing users to easily exploit a wide range of techniques in learning, mining, visualization, IE, etc.

Finally, an added benefit of integration with Python is that Magellan is situated in a powerful interactive scripting environment that users can use to prototype code to “patch” the system.

**Challenges:** Realizing the above novelties raises major challenges. First, it turns out that developing effective howto guides, even for very simple EM scenarios such as applying supervised learning to match, is already quite difficult and complex, as we will show in Section 4. Second, developing tools to support these guides is equally difficult. In particular, current EM work may have dismissed many steps in the EM pipeline as engineering. But here we show that many such steps (e.g., loading the data, sampling and labeling, debugging, etc.) do raise difficult research challenges.

Finally, while most current EM systems are stand-alone monoliths, Magellan is designed to be placed within an “ecosystem” and is expected to “play well” with others (e.g., other Python packages). We distinguish this by saying that current EM systems are “closed-world systems” whereas Magellan is an “open-world system”, because it relies on many other systems in the eco-system in order to provide the fullest amount of support to the user doing EM. It turns out that building open-world systems raises non-trivial challenges, such as designing the right data structures and managing metadata, as we discuss in Section 5.

In this paper we have taken the first steps in addressing the above challenges. We have also built and evaluated Magellan 0.1 in several real-world settings (e.g., at WalmartLabs, Johnson Control Inc., Marshfield Clinic) and in data science

<!-- Start of picture text -->
Table A Table B<br>Name City State Name City State Matches<br>a1  Dave Smith Madison WI b1 David D. Smith Madison WI (a1, b1)<br>a2  Joe Wilson San Jose CA b2 Daniel W. Smith Middleton WI (a3, b2)<br>a3   Dan Smith Middleton WI<br><!-- End of picture text -->

#### **Figure 1: An example of matching two tables.**

classes at UW-Madison. In summary, we make the following contributions:

- We argue that far more efforts should be devoted to building EM systems, to significantly advance the field.

- We discuss four limitations that prevent current EM systems from being used extensively in practice.

- We describe the Magellan system, which is novel in several important aspects: how-to guides, tools to support all steps of the EM pipeline, tight integration with the Python data eco-system, easy access to an interactive scripting environment, and open world vs. closed world systems.

- We describe significant challenges in realizing Magellan, including the novel challenge of designing openworld systems (that operate in an eco-system).

- We describe extensive experiments with 44 students and real users at various organizations that show the utility of Magellan, including improving the accuracy of an EM system in production.

A shorter version of this technical report has been published in VLDB-2016. Magellan will be released at the website sites.google.com/site/anhaidgroup/projects/magellan in Summer 2016, to serve research, development, and practical uses. Finally, the ideas underlying Magellan can potentially be applied to other types of DI problems (e.g., IE, schema matching, data cleaning, etc.), and an effort has been started to explore this direction and to foster an eco-system of opensource DI tools (see Magellan’s website).

## **2. THE CASE FOR ENTITY MATCHING MANAGEMENT SYSTEMS**

## **2.1 Entity Matching**

This problem, also known as record linkage, data matching, etc., has received much attention in the past few decades [16, 22]. A common EM scenario finds all tuple pairs ( _a, b_ ) that match, i.e., refer to the same real-world entity, between two tables _A_ and _B_ (see Figure 1). Other EM scenarios include matching tuples within a single table, matching into a knowledge base, matching XML data, etc. [16].

Most EM works have developed matching algorithms, exploiting rules, learning, clustering, crowdsourcing, among others [16, 22]. The focus is on improving the matching accuracy and reducing costs (e.g., run time). Trying to match all pairs in _A × B_ often takes very long. So users often employ heuristics to remove obviously non-matched pairs (e.g., products with different colors), in a step called _blocking_ , before matching the remaining pairs. Several works have studied this step, focusing on scaling it up to large amounts of data (see Section 8).

2

|**Name**|**Affiliation**|**Scenarios**|**Blocking**|**Matching**|**Exploration,**<br>**cleaning**|**User**<br>**interface**|**Language**|**Open**<br>**source**|<br> <br>**Scaling**|
|---|---|---|---|---|---|---|---|---|---|
|**Active Atlas**|University of<br>Southern<br>California|Single table, two<br>tables|Hash-based|ML-based (decision<br>tree)|<br>No|GUI,<br>commandline|Java|No|No|
|**BigMatch**|US Census<br>Bureau|Single table, two<br>tables|Attribute<br>equivalence, rule-<br>based|Not supported|No|Commandline|C|No|Yes (supports<br>parallelism on<br>a single node)|
|**D-Dupe**|University of<br>Maryland|Single table, two<br>tables|Attribute<br>equivalence|Relational clustering||GUI|C#|No|No|
|**Dedoop**|University of<br>Leipzig|Single table|Attribute<br>equivalence,<br>sorted<br>neighborhood|ML-based (decision<br>tree, logistic<br>regression, SVM<br>etc.)|<br>No|GUI|Java|No|Yes (Hadoop)|
|**Dedupe**|Datamade|Single table, two<br>tables<br>|Canopy clustering,<br>predicate-based|<br>Agglomerative<br>hierarchical<br>clustering-based|Browsing,<br>statistics, basic<br>transformation,<br>cleaning certain<br>attribute types|<br>Commandline|Python|Yes|Yes|
|**DuDe**|University of<br>Potsdam|Single table, two<br>tables|Sorted<br>neighborhood|Rule-based|Statistics|Commandline|Java|Yes|No|
|**Febrl**|Australian<br>National<br>University|Single table, two<br>tables|Full index,<br>blocking index,<br>sorting index,<br>suffixarray index,<br>qgram index,<br>canopy index,<br>stringmapindex|Fellegi-Sunter,<br>optimal threshold,<br>k-means,<br>FarthestFirst, SVM,<br>TwoStep|<br>Browsing,<br>statistics, basic<br>transformation,<br>cleaning certain<br>attribute types|<br>GUI,<br>commandline|Python|Yes|No|
|**FRIL**|Emory<br>University|Single table, two<br>tables|Attribute<br>equivalence,<br>sorted<br>neighborhood|Expectation<br>maximization|Basic<br>transformation,<br>cleaning certain<br>attribute types|<br>GUI|Java|Yes|Yes (supports<br>parallelism on<br>a single node)|
|**MARLIN**|University of<br>Texas at Austin||Canopy clustering|<sup>ML-based (decision</sup><br>tree,SVM)|||||No|
|**Merge**<br>**Toolbox**|University of<br>Duisburg-Eissen|<br>Single table, two<br>tables|Attribute<br>equivalence,<br>canopyclustering|Probabilistic,<br>expectation<br>maximization|No|GUI|Java|No|No|
|**NADEEF**|Qatar Computing<br>Research<br>Institute|<br>Single table, two<br>tables||Rule-based|No|GUI|Java|No|No|
|**OYSTER**|University of<br>Arkansas|Single table, two<br>tables|Attribute<br>equivalence|Rule-based|Statistics|Commandline|Java|Yes|No|
|**pydedupe**|GPoulter<br>(GitHub<br>username)|Single table, two<br>tables|Attribute<br>equivalence|ML-based, rule-<br>based|Browsing,<br>statistics, basic<br>transformation,<br>cleaning certain<br>data types|<br>Commandline|Python|Yes|No|
|**RecordLinkag**<br>**e**|Institute of<br>Medical<br>Biostatistics,<br>Germany|Single table, two<br>tables|Attribute<br>equivalence|ML-based,<br>probabilistic|Browsing,<br>statistics, basic<br>transformation,<br>cleaning certain<br>attribute types|<br>Commandline|R|Yes|No|
|**SERF**|Stanford<br>University|Single table||R-Swoosh algorithm|<br>No|Commandline|Java|No|No|
|**Silk**|Free University<br>of Berlin|RDF data||Rule-based|Browsing, basic<br>transformation|<br>GUI|Java|Yes|Yes (supports<br>parallelism on<br>a single node,<br>Hadoop)|
|**TAILOR**<br>**WHIRL**|Purdue<br>University<br>William Cohen|Single table, two<br>tables|Attribute<br>equivalence,<br>sorted<br>neighborhood|Probabilisitic,<br>clustering, hybrid,<br>induction<br>Vector space model|No<br>|GUI<br>Commandline|Java<br>C++|No<br>No|No<br>No|

**Table 1: Characteristics of 18 non-commercial EM systems.**

## **2.2 Current Entity Matching Systems**

In contrast to the extensive effort on matching algorithms (e.g., 96 papers were published on this topic in 2009-2014

alone, in SIGMOD, VLDB, ICDE, KDD, and WWW), there has been relatively little work on building EM systems. As of early 2016 we counted 18 major non-commercial systems

3

(e.g., D-Dupe, DuDe, Febrl, Dedoop, Nadeef), and 15 major commercial ones (e.g., Tamr, Data Ladder, Informatica Data Quality). In what follows we examine these two types of systems in detail.

### _2.2.1 Non-Commercial EM Systems_

Table 1 summarizes the characteristics of 18 non-commercial systems (see [16] for a discussion of such systems up to 2012). Empty cells mean reliable information cannot be gleaned from the documentation and system examination. This table shows that

- The systems focus on the scenarios of matching within a single table or across two tables.

- They provide a wide range of methods for the wellknown blocking and matching steps, but no guidance on how to select appropriate blockers and matchers.

- Eight systems provide limited data exploration capabilities (e.g., browsing, showing statistics about the data) and cleaning capabilities (mostly ways to perform relatively simple transformations such as regexbased ones and to clean certain common attributes such as person names). No system provides support for less well-known but critical steps such as debugging, sampling, and labeling.

- No system provides how-to guides that tell users how to do EM, step by step. And no system makes a distinction between the development stage and the production stage (i.e., guiding users to develop a good EM workflow in the development stage and then execute the workflow in the production stage).

- Less than half of the systems are open source. No system provides any easy interfacing with data science stacks (and is not intentionally designed to interface with such stacks).

- Thirteen systems are written in languages such as C, C#, C++, and Java, and thus are not situated in a powerful scripting environment that facilitates rapid and iterative experimentation (e.g., examining the effect of a data cleaning operation, trying out a different blocker or matcher).

- About half of the systems provide just commandline interfaces, while the remaining half also provide GUIs. A few systems provide limited scaling capabilities.

### _2.2.2 Commercial EM Systems_

We compiled a list of 15 commercial EM systems from our experience working in industry, and from examining quarterly reports such as “The Forrester Wave: Data Quality Solutions” and other trade literature. Tables 2-3 summarize the characteristics of these systems. Again, the empty cells in the tables mean reliable information cannot be gleaned from the documentation and system examination.

Table 2 summarizes the general characteristics of the commercial systems. It shows that

- Five systems focus exclusively on EM. The remaining ten systems provide EM as a part of data integration or cleaning pipelines.

- The systems focus on the scenarios of matching within a single table or across two tables. Unlike non-commercial systems, these systems have very sophisticated GUI or Web-based user interfaces.

- There is no how-to guide that tells users how to do EM, step by step. Instead, the vendors sell consulting services (sometimes called “data stewarding”) that presumably help users use the systems. Seven systems make no distinction between the development stage and the production stage. For the remaining eight systems we cannot reliably tell from the documentation, but they do not seem to make such a distinction either.

- Many systems use languages such as C++ and Java. As far as we can tell, no system (except GraphLab Create) is situated in a powerful scripting environment for rapid and iterative experimentation.

- No system is open source and designed to interface well with tools in a data science stack.

Table 3 summarizes the support for the entire EM pipeline in these systems. It shows that

- These systems support far more types of input data (e.g., relational tables, JSON, CSV, XML) than the non-commercial systems.

- There seems to be more support for data exploration and cleaning (compared to non-commercial systems), though still limited. Data exploration is typically accomplished via GUIs that display statistics about the data (e.g., the percentage of missing values of an attribute). Many systems provide tools to clean common kinds of attributes (e.g., addresses, phone numbers, person names). But powerful general-purpose data cleaning tools are typically missing.

- Interestingly, these systems do not seem to provide as many different types of blocking and matching as the non-commercial systems. For example, the most common type of supported blocking is attribute equivalence, and the most common type of supported matching is rule-based. It is possible that these systems need to scale EM to very large amounts of data and so they intentionally limit the set of blocking and matching techniques considered for now, to ensure scalability. Indeed, virtually all systems provide capabilities to scale, using Hadoop and Spark.

- There is very limited or no support for other critical steps of the EM pipeline, such as sampling, debugging, and labeling. For example, there is no support for debugging blockers, and support for debugging matchers is typically limited to showing which EM rule fires on a given tuple pair.

We now describe a few selected commercial systems, specifically SAS Data Quality, Informatica Data Quality, DataMatch, and Tamr.

**SAS Data Quality:** This system (henceforth SAS for short) provides EM as a part of their data quality pipeline. SAS focuses on the scenarios of matching within a single

4

||**Purpose and how EM fits in**|**Supported EM**<br>**scenarios**|**Main user**<br>**interface**|**Distinction between**<br>**dev. andprod. stages**|**Language**|**Scripting**<br>**environment**|
|---|---|---|---|---|---|---|
|**DataMatch**<br>**from Data**<br>**Ladder**|Data cleaning, data matching.<br>EM forms the core of their<br>solution|Multiple tables|GUI|No||No|
|**Dedupe.io**|Record linkage, deduplication.<br>EM forms the core of their<br>solution|Single table, two tables|Web-based|No||No|
|**FuzzyDupes**|Duplicate detection, data<br>cleaning. EM forms the core of<br>their solution|Single table, two tables|GUI|No||No|
|**Graphlab**<br>**Create**|EM is offered as a service on<br>top of their GraphLab platform|Single table, two tables,<br>linking records to a KB|Web-based||C++|Yes|
|**IBM**<br>**InfoSphere**|Customer data analytics. EM is<br>supported by a component<br>(BigMatch) in the product|Single table, two tables|Web-based||Java|No|
|**Informatica**<br>**Data Quality**|Improve data quality. EM forms<br>a part of data quality pipeline|<br>Single table, two tables|GUI|||No|
|**LinkageWiz**|Data matching and data<br>cleaning. EM forms the core of<br>their solution|Single table, two tables|GUI|No||No|
|**Oracle**<br>**Enterprise**<br>**Data Quality**|Improve data quality. EM forms<br>a part of data quality pipeline|<br>Single table, two tables|GUI|||No|
|**Pentaho Data**<br>**Integration**|<br>ETL, data integration. EM<br>forms a part of ETL/data<br>integration pipe line|Single table, two tables|GUI||Java|No|
|**SAP Data**<br>**Services**|Improve data quality, data<br>integration. EM forms a part of<br>data integration pipeline|Single table, two tables|GUI|No|||
|**SAS Data**<br>**Quality**|Improve data quality. EM forms<br>a part of data quality pipeline|<br>Single table, multiple<br>tables|Web-based|||Limited support|
|**Strategic**<br>**Matching**|Data matching and data<br>cleaning. EM forms the core of<br>their solution|Single table, two tables|GUI|No||No|
|**Talend Data**<br>**Quality**|Improve data quality. EM forms<br>a part of data quality pipeline|<br>Single table, two tables|GUI|||No|
|**Tamr**|Data curation. EM forms a part<br>of data curationpipeline|Multiple tables|Web-based|No|Java|No|
|**Trillium Data**<br>**Quality**|<br>Improve data quality. EM forms<br>a part of data quality pipeline|<br>Single table, multiple<br>tables|GUI|||No|

**Table 2: Characteristics of 15 commercial EM systems (Part 1).**

table or across multiple tables. The EM workflow supported in SAS consists of five major steps.

First, the user loads the data into SAS. SAS supports various data formats and sources, such as Excel, CSV, XML, delimited text files, relational databases, and HDFS.

Second, the user explores the loaded data. SAS lets the user perform pattern analysis, column analysis, and domain analysis. In pattern analysis the user can verify if the data values in an attribute match the expected pattern (e.g., 9- digits for SSN, 10-digits for phone numbers), and visualize the distribution and frequency for various patterns, e.g., how

many phone numbers were of the form (xxx) xxx-xxxx). In column analysis, the user can explore various statistics (e.g., cardinality, number of missing values, range, min, mean, median) of a column in a table. In domain analysis, the user can verify if the data conforms to the expected or accepted data values and ranges (e.g., age is between 0 and 150 years).

Third, the user cleans and standardizes the data. In cleaning, the user can fix capitalization in data values, remove punctuations, break a “full name” column into “first name” and “last name” columns by specifying a delimiter, etc. In standardization, the user specifies that an attribute is of the

5

||**Supported data**<br>**formats/sources**|**Data**<br>**exploration**<br>**support**|<br>**Data**<br>**cleaning**<br>**support**|**Down**<br>**sampling**<br>**input**<br>**table(s)**|**Blocking**|**Support to**<br>**combine**<br>**multiple**<br>**blockers**|**Debugging**<br>**blocker**<br>**output**|<br>**Labeling**<br>**data**|<br>**Matching**|**Debugging**<br>**matcher**<br>**output**|<br>**Scaling**|
|---|---|---|---|---|---|---|---|---|---|---|---|
|**DataMatch from**<br>**Data Ladder**|<br>Relational databases,<br>XLS, DB2, CSV,<br>delimited text files|Browsing,<br>statistics|Yes|No|Not supported|No|No|No|Rule-based|Limited<br>support|Yes|
|**Dedupe.io**|Relational databases<br>(Postgres), CSV, XLS,||||Canopy<br>clustering,<br>predicate-based<br>blocking|No|No|Yes|Clustering-<br>based (AHC)|<br>Limited<br>support|Yes|
|**FuzzyDupes**|Relational databases,<br>XLS, CSV, delimited<br>text files|||No||No|No|No|||Yes|
|**Graphlab Create**|<br>Relational databases,<br>CSV, Pandas<br>dataframes , HDFS,<br>Amazon S3, JSON|Browsing,<br>statistics|||Attribute<br>equivalence|No|||Clustering-<br>based (KNN)||Yes (Hadoop,<br>Spark)|
|**IBM InfoSphere**|<br>Relational databases,<br>XLS, delimited text<br>files, XML, JSON,<br>HDFS, text files|Browsing,<br>statistics|Yes||Attribute<br>equivalence,<br>blocking based<br>on first 3<br>characters,<br>phonetic codes||||Rule-based||Yes (Hadoop)|
|**Informatica**<br>**Data Quality**|Relational databases,<br>CSV, excel, XML,<br>delimited text files,<br>HDFS|Browsing,<br>statistics|Yes|No|Attribute<br>equivalence|No|No|No|Rule-based||Yes|
|**LinkageWiz**|XLS, delimited text<br>files,SPSS|Browsing,<br>statistics|Yes|No|Attribute<br>equivalence|No|No|No|Rule-based|Limited<br>support||
|**Oracle**<br>**Enterprise Data**<br>**Quality**|<br>Relational databases,<br>XLS, delimited text files|<br>Browsing,<br>statistics|Yes|No||No|No|No|Rule-based|Limited<br>support|Yes (Hadoop,<br>Hive, HBase,<br>Pig, Sqoop,<br>Spark)|
|**Pentaho Data**<br>**Integration**|Relation databases,<br>CSV, XML, JSON,<br>MongoDB, NuoDB,<br>Couchbase, Avro|Browsing,<br>statistics|Yes||||No||Rule-based||Yes (Hadoop,<br>Spark, Mongo<br>DB, Splunk,<br>Cassandra)|
|**SAP Data**<br>**Services**|Relational databases,<br>CSV, XLS, JSON,<br>XML,HDFS|Browsing,<br>statistics|Yes|No|Attribute<br>equivalence|No|No|No|Rule-based||Yes (Hadoop,<br>Spark)|
|**SAS Data**<br>**Quality**|Relational databases,<br>XLS and delimited text<br>files,XML|Browsing,<br>statistics|Yes||Not supported||||Hash-based||Yes (Hadoop)|
|**Strategic**<br>**Matching**|Relational databases<br>(SQL server), MS<br>Access,SAS|Browsing,<br>statistics|Yes|No|||No|No|Rule-based|Limited<br>support||
|**Talend Data**<br>**Quality**|Relational databases,<br>CSV, XLS, XML,<br>JSON,EBCDIC|Browsing,<br>statistics|Yes|No|Attribute<br>equivalence|No|No|No|Rule-based|Limited<br>support|Yes (Hadoop,<br>Spark)|
|**Tamr**|Relational databases,<br>JSON, XML, YAML,<br>RDF, HDFS, Hive,<br>Amazon/redshift,<br>Google cloud storage,<br>MongoDB, Cloudant,<br>Cassandra,CSV,XLS|||No|Modified k-<br>means|No|No|Yes|Rule-based|Limited<br>support|Yes|
|**Trillium Data**<br>**Quality**|Relational databases,<br>CSV, XLS, JSON,<br>HDFS,NoSQL|Browsing,<br>statistics|Yes||||||Rule-based||Yes (Hadoop,<br>Spark)|

**Table 3: Characteristics of 15 commercial EM systems (Part 2).**

type “name”, “address”, “phone”, etc. and SAS makes sure that names are capitalized consistently, addresses use “st.” as an abbreviation for street names, etc.

Fourth, the user performs hash-based matching in a single table or across multiple tables. Specifically, the user first selects the attributes (say _a_ 1, _a_ 2, _a_ 3) to consider for matching. For every tuple _t_ , SAS will then generate a hash code, _h_ ( _t_ ), which is a concatenation of multiple smaller hash codes, one per attribute, i.e., _h_ ( _t_ ) = _h_ ( _t.a_ 1)! _h_ ( _t.a_ 2)! _h_ ( _t.a_ 3), where ! is the concatenating delimiter.

SAS generates the hash code per attribute by taking two

inputs from the user: (a) _type_ value for the attribute from a pre-defined set, comprising standard types such as name, address, organization, date, zip, and (b) a sensitivity value for the attribute telling SAS how sensitive the hashing function should be to variations in values (e.g., a low sensitivity will result in same hash code for Rob, Robert, Bob, Bobby; a moderate sensitivity will result in same hash code for Rob and Robert, but a different hash code for Bob and Bobby; a high sensitivity will result in different hash codes for each of them).

Finally, after the hash codes have been generated for each

6

tuple in a table (or multiple tables), SAS will show the tuples grouped into clusters, each cluster having tuples with the same hash code. The user then consolidates the data by taking one of the three actions of deleting (i.e., physically deleting duplicate tuples), merging (i.e., keeping the best information across multiple tuples), or retaining all the tuples.

**Informatica Data Quality:** This system provides EM as a part of its data quality pipeline. Specifically, it supports matching within a single table or across two tables. The supported EM workflow consists of six steps.

First, the user loads the data into the system. The system supports various data formats such as CSV, Excel, XML, delimited text files etc.

Second, the user explores the data to identify attributes to use for blocking and matching. The system provides tools to analyze individual attributes and explore various statistics about the attributes.

Third, the user cleans and standardizes the data. Specifically, the user can fix variations in format or spelling, remove punctuations, fix capitalization etc. Further, the system also provides support to standardize certain attribute types like address, phone number etc.

Fourth, the user performs blocking by selecting an attribute to be used as a blocking key. Records with same blocking key are grouped together.

Next, the user will perform matching within each group. Specifically, the system supports 4 types of matchers: Hamming distance, edit distance, Jaro distance, and bigram. The user needs to specify which matchers to use, along with a matching threshold and weights for different matchers. Record pairs whose aggregate score is greater than or equal to the matching threshold are considered duplicates. The system groups the matching record pairs into clusters.

Finally, the user examines the clusters of records and decides to either consolidate the duplicate records into a master record or delete the duplicate records.

**DataMatch:** DataMatch from Data Ladder provides a software suite for data cleansing, matching, and deduplication. Entity matching is the core of their solution. Specifically, the tool supports deduplicating a single table or matching multiple tables. The matching workflow consists of the following six steps: (1) loading the data, (2) profiling, (3) cleaning and standardizing, (4) matching, (5) viewing and consolidating the results, and (6) exporting the results.

The user begins by loading the data into the tool (the tool supports various data formats/sources such as XLS, SQL server, MySQL, MS Access, CSV, DB2, and delimited text file). Next, the user can explore the data to assess the data quality and get some useful statistics (e.g., missing values, presence of non-printable characters, mean, median, mode). Next, the user can clean and standardize the data. The tool provides support for basic transformations such as making strings uppercase/lowercase/proper case, removing non-printable characters, removing characters specified by the user, and cleaning email using predefined regular expressions. Further, the tool also provides support to standardize certain attribute types such as person names, address, etc.

After cleaning, the user will perform matching. The tool supports only rule-based matching. Specifically, the user will specify the features (using a predefined list of similarity functions) to be computed for the attributes from the

tables, and provide a matching threshold. Tuple pairs with the aggregate score greater than or equal to the matching threshold are considered matches. Next, the user can view and consolidate the matched tuple pairs. The user can manually review and clean the matches by flagging tuple pairs as non-matches.

Next, the matched tuple pairs are clustered by the system into groups, where all tuples in a group match and tuples across groups do not. Next, the user can specify how the group should be merged to form a canonical tuple. Specifically, for each attribute the user can specify whether the longest string should be taken, the average value (in the case of numerical values) should be taken, etc. Also, the user can control this decision per tuple pair.

Finally, the user can export the results. The tool provides exporting the results to various file formats/sinks such as XLS, SQL server, MySQL, MS Access, CSV, DB2, and delimited text file.

**Tamr:** This system has entity matching as a component in a data curation pipeline. This EM component effectively does deduplication and merging: given a set of tuples _D_ , clusters them into groups of matching tuples, and then merges each group into a super tuple.

Toward the above goal, Tamr starts by performing blocking on the set of tuples _D_ . Specifically, it creates a set of categories, then use some relatively inexpensive similarity measure to assign each tuple in _D_ to one or several categories. Only tuples within each category will be matched against one another.

Next, Tamr obtains a set of tuple pairs and asks users to manually label them as matched / non-matched. Tamr takes care to ensure that there are a sufficient number of matched tuple pairs in this set. Next, Tamr uses the labeled data to learn a set of matching rules. These rules use the similarity scores among the attributes of a tuple pair, or the probability distributions of attribute similarities for matching and non-matching pairs (these probabilities in turn are learned using a Naive Bayes classifier).

Next, the matching rules are applied to find matching tuple pairs. Tamr then runs a correlation clustering algorithm that uses the matching information to group tuples into matching group. Finally, all tuples within each group are consolidated using user-defined rules to form a super tuple.

## **2.3 Key Limitations of Current Systems**

Overall, we found that commercial EM systems are better than non-commercial EM systems in terms of support for the types of input data, user interfaces, data exploration and cleaning, and scaling. They appear less powerful than the non-commercial ones in terms of the types of supported blocking and matching techniques.

Both types of systems however suffer from the following four major problems that we believe prevent these systems from being used widely in practice:

**1. Systems Do Not Cover the Entire EM Pipeline:** When performing EM users often must execute many steps, e.g., blocking, matching, exploration, cleaning, extraction (IE), debugging, sampling, labeling, etc. Current systems provide support for only a few steps in this pipeline, while ignoring less well-known yet equally critical steps.

7

For example, all 33 systems that we have examined provide support for blocking and matching. Twenty systems provide limited support for data exploration and cleaning. There is no meaningful support for any other steps (e.g., debugging, sampling, etc.). Even for blocking the systems merely provide a set of blockers that users can call; there is no support for selecting and debugging blockers, and for combining multiple blockers.

**2. Difficult to Exploit a Wide Range of Techniques:** Practical EM often requires a wide range of techniques, e.g., learning, mining, visualization, data cleaning, IE, SQL querying, crowdsourcing, keyword search, etc. For example, to improve matching accuracy, a user may want to clean the values of attribute “Publisher” in a table, or extract brand names from “Product Title”, or build a histogram for “Price”. The user may also want to build a matcher that uses learning, crowdsourcing, or some statistical techniques.

Current EM systems do not provide enough support for these techniques, and there is no easy way to do so. Incorporating all such techniques into a single system is extremely difficult. But the alternate solution of just moving data among a current EM system and systems that do cleaning, IE, visualization, etc. is also difficult and time consuming. A fundamental reason is that most current EM systems are stand-alone monoliths that are not designed from the scratch to “play well” with other systems. For example, many current EM systems were written in C, C++, C#, and Java, using proprietary data structures. Since EM is often iterative, we need to repeatedly move data among these EM systems and cleaning/IE/etc systems. But this requires repeated reading/writing of data to disk followed by complicated data conversion.

#### **3. Difficult to Write Code to “Patch” the System:**

In practice users often have to write code, either to implement a lacking functionality (e.g., to extract product weights, or to clean the dates), or to tie together system components. It is difficult to write such code correctly in “one shot”. Thus ideally such coding should be done using an interactive scripting environment, to enable rapid prototyping and iteration. This code often needs access to the rest of the system, so ideally the system should be in such an environment too. Unfortunately only 5 out of 33 systems provide such settings (using Python and R).

**4. Little Guidance for Users on How to Match:** In our experience this is by far the most serious problem with using current EM systems in practice. In many EM scenarios users simply do not know what to do: how to start, what to do next? Interestingly, even the simple task of taking a sample and labeling it (to train a learning-based matcher) can be quite complicated in practice, as we show in Section 4.3. Thus, it is not enough to just build a system consisting of a set of tools. It is also critical to provide step-by-step guidance to users on how to use the tools to handle a particular EM scenario. No EM system that we have examined provides such guidance.

## **2.4 Entity Matching Management Systems**

To address the above limitations, we propose to build a new kind of EM systems. In contrast to current EM systems, which mostly provide a set of implemented matchers/blockers, these new systems are far more advanced.

First and foremost, they seek to handle a wide variety of EM scenarios. These scenarios can use very different EM workflows. So it is difficult to build a single system to handle all EM scenarios. Instead, we should build a set of systems, each handling a well-defined set of similar EM scenarios. Each system should target the following goals:

1. **How-to Guide:** Users will have to be “in the loop”. So it is critical that the system provides a how-to guide that tells users what to do and how to do it.

2. **User Burden:** The system should minimize the user burden. It should provide a rich set of tools to help users easily do each EM step, and do so for all steps of the EM pipeline, not just matching and blocking. Special attention should be paid to debugging, which is critical in practice.

3. **Runtime:** The system should minimize tool runtimes and scale tools up to large amounts of data.

4. **Expandability:** It should be easy to extend the system with any existing or future techniques that can be useful for EM (e.g., cleaning, IE, learning, crowdsourcing). Users should be able to easily “patch” the system using an interactive scripting environment.

Of these goals, “expandability” deserves more discussion. If we can build a single “super-system” for EM, do we need expandability? We believe it is very difficult to build such a system. First, it would be immensely complex to build just an initial system that incorporates all of the techniques mentioned in Goal 4. Indeed, despite decades of development, today no EM system comes close to achieving this.

Second, it would be very time consuming to maintain and keep this initial system up-to-date, especially with the latest advances (e.g., crowdsourcing, deep learning).

Third, and most importantly, a generic EM system is unlikely to perform equally well for multiple domains (e.g., biomedicine, social media, payroll). Hence we often need to extend and customize it to a particular target domain, e.g., adding a data cleaning package specifically designed for biomedical data (written by biomedical researchers). For the above three reasons, we believe that EM systems should be fundamentally expandable.

Clearly, systems that target the above goals seek to _manage_ all aspects of the end-to-end EM process. So we refer to this kind of systems as _entity matching management systems (EMMSs)_ . Building EMMSs is difficult, long-term, and will require a new kind of architecture compared to current EM systems. In the rest of this paper we describe Magellan, an attempt to build such an EMMS.

## **3. THE MAGELLAN APPROACH**

Figure 2 shows the Magellan architecture. The system targets a set of EM scenarios. For each EM scenario it provides a how-to guide. The guide proposes that the user solve the scenario in two stages: development and production.

In the development stage, the user seeks to develop a good EM workflow (e.g., one with high matching accuracy). The guide tells the user what to do, step by step. For each step the user can use a set of supporting tools, each of which is in turn a set of Python commands. This stage is typically done using data samples. In the production stage, the guide tells

8

<!-- Start of picture text -->
Facilities for Lay Users<br>GUIs, wizards, …<br>Power Users<br>EM  Development Stage  Production Stage<br>Scenarios<br>Supporting tools  EM Supporting tools<br>How-to (as Python commands)  Workflow (as Python commands)<br>Guides<br>Data samples  Original data<br>Python Interactive Environment<br> Script Language<br>Data Analysis Stack  Big Data  Stack<br>pandas,  scikit-learn, matplotlib,  PySpark, mrjob, Pydoop,<br>…     …<br><!-- End of picture text -->

**Figure 2: The Magellan architecture.**

the user how to implement and execute the EM workflow on the entirety of data, again using a set of supporting tools. Both stages have access to the Python script language and interactive environment (e.g., iPython). Further, tools for these stages are built on top of the Python data analysis stack and the Python Big Data stack, respectively. Thus, Magellan is an “open-world” system, as it often has to borrow functionalities (e.g., cleaning, extraction, visualization) from other Python packages on these stacks.

Finally, the current Magellan is geared toward power users (who can program). We envision that in the future facilities for lay users (e.g., GUIs, wizards) can be laid on top (see Figure 2), and lay user actions can be translated into sequences of commands in the underlying Magellan.

In the rest of this section, we describe EM scenarios, workflows, and the development and production stages. Section 4 describes the how-to guides, and Section 5 describes the challenges of designing Magellan as an open-world system.

## **3.1 EM Scenarios and Workflows**

We classify EM scenarios along four dimensions:

- **Problems:** Matching two tables; matching within a table; matching a table into a knowledge base; etc.

- **Solutions:** Using learning; using learning and rules; performing data cleaning, blocking, then matching; performing IE, then cleaning, blocking, and matching; etc.

- **Domains:** Matching two tables of biomedical data; matching e-commerce products given a large product taxonomy as background knowledge; etc.

- **Performance:** Precision must be at least 92%, while maximizing recall as much as possible; both precision and recall must be at least 80%, and run time under four hours; etc.

An EM scenario can constrain multiple dimensions, e.g., matching two tables of e-commerce products using a rulebased approach with desired precision of at least 95%.

Clearly there is a wide variety of EM scenarios. So we will build Magellan to handle a few common scenarios, and then extend it to more similar scenarios over time. Specifically, for now we will consider the three scenarios that match two

given relational tables _A_ and _B_ using (1) supervised learning, (2) rules, and (3) learning plus rules, respectively. These scenarios are very common. In practice, users often try Scenario 1 or 2, and if neither works, then a combination of them (Scenario 3).

**EM Workflows:** As discussed earlier, to handle an EM scenario, a user often has to execute many steps, such as cleaning, IE, blocking, matching, etc. The combination of these steps form an _EM workflow_ . Figure 9 shows a sample workflow (which we explain in detail in Section 4.6).

## **3.2 The Development vs. Production Stages**

From our experience with real-world users’ doing EM, we propose that the how-to guide tell the user to solve the EM scenario in two stages: _development_ and _production_ . In the development stage the user tries to find a good EM workflow, e.g., one with high matching accuracy. This is typically done using data samples. In the production stage the user applies the workflow to the entirety of data. Since this data is often large, a major concern here is to scale up the workflow. Other concerns include quality monitoring, logging, crash recovery, etc. The following example illustrates these two stages.

Example 1. _Consider matching two tables A and B each having 1M tuples. Working with such large tables will be very time consuming in the development stage, especially given the iterative nature of this stage. Thus, in the development stage the user U starts by sampling two smaller tables A_<sup>_′_</sup> _and B_<sup>_′_</sup> _from A and B, respectively. Next, U performs blocking on A_<sup>_′_</sup> _and B_<sup>_′_</sup> _. The goal is to remove as many obviously nonmatched tuple pairs as possible, while minimizing the number of matching pairs accidentally removed. U may need to try various blocking strategies to come up with what he or she judges to be the best._

_The blocking step can be viewed as removing tuple pairs from A_<sup>_′_</sup> _×B_<sup>_′_</sup> _. Let C be the set of remaining tuple pairs. Next, U may take a sample S from C, examine S, and manually write matching rules, e.g., “If titles match and the numbers of pages match then the two books match”. U may need to try out these rules on S and adjust them as necessary. The goal is to develop matching rules that are as accurate as possible._

_Once U has been satisfied with the accuracy of the matching rules, the production stage begins. In this stage, U executes the EM workflow that consists of the developed blocking strategy and matching rules on the original tables A and B. To scale, U may need to rewrite the code for blocking and matching to use Hadoop or Spark. 2_

As described, these two stages are very different in nature: one goes for accuracy and the other goes for scaling (among others). Consequently, they will require very different sets of tools. We now discuss developing tools for these stages.

**Development Stage on a Data Analysis Stack:** We observe that what users try to do in the development stage is very similar in nature to data analysis tasks, which analyze data to discover insights. Indeed, creating EM rules can be viewed as analyzing (or mining) the data to discover accurate EM rules. Conversely, to create EM rules, users also often have to perform many data analysis tasks, e.g., cleaning, visualizing, finding outliers, IE, etc.

9

As a result, if we are to develop tools for the development stage in isolation, within a stand-alone monolithic system, as current work has done, we would need to somehow provide a powerful data analysis environment, in order for these tools to be effective. This is clearly very difficult to do.

So instead, we propose that tools for the development stage be developed on top of an open-source data analysis stack, so that they can take full advantage of all the data analysis tools already (or will be) available in that stack. In particular, two major data analysis stacks have recently been developed, based on R and Python (new stacks such as the Berkeley Data Analysis Stack are also being proposed). The Python stack for example includes the general-purpose Python language, numpy and scipy packages for numerical/array computing, pandas for relational data management, scikit-learn for machine learning, among others. More tools are being added all the time, in the form of Python packages. By Oct 2015, there were 490 packages available in the popular Anaconda distribution. There is a vibrant community of contributors to continuously improve this stack.

For Magellan, since our initial target audience is the IT community, where we believe Python is more familiar, we have been developing tools for the development stage on the Python data analysis stack.

**Production Stage on a Big Data Stack:** In a similar vein, we propose that tools for the production stage, where scaling is a major focus, be developed on top of a Big Data stack. Magellan uses the Python Big Data stack, which consists of many software packages to run MapReduce (e.g., Pydoop, mrjob), Spark (e.g., PySpark), and parallel and distributed computing in general (e.g., pp, dispy).

**Expandability Revisited:** We are now in a position to discuss how Magellan addresses the expandability requirement outlined in Section 2.4. Current EM systems address expandability in two ways: adding external libraries or moving data among a set of stand-alone systems (e.g., an EM system, an IE system, a visualization system, etc.).

Both methods are problematic. To add an external library we need to write extra code to convert between the data structures used by the system and the library. This is time consuming and may not even be feasible if we do not have access to the system code. Moving data repeatedly among a set of stand-alone systems is very cumbersome as it requires repeatedly writing data to disk, reading data from disk, and converting between the various data formats.

As discussed in Section 2.3, the root of these problems is that most current EM systems are not designed from the scratch to support expandability. In contrast, Magellan assumes that there is already an eco-system of “systems” (in form of Python packages) that have been designed to expand (i.e., “play well” with one another) and that Magellan will have to be in that eco-system and to “play well” too.

In sum, the Magellan solution for expandability is to design the system such that it can be easily “plugged” into an existing and expanding data management eco-system, and that it can combine well with tools in this eco-system.

As an aside, this approach also brings the non-trivial benefit that we are filling in “gaps” in the Python data management eco-system. This eco-system is important because more and more users are using its tools to analyze data, but so far good EM tools (and good data integration tools

1. Load tables A and B into Magellan. Downsample if necessary.

2. Perform blocking on the tables to obtain a set of candidate tuple pairs C.

3. Take a random sample S from C and label pairs in S as matched / non-matched.

4. Create a set of features then convert S into a set of feature vectors H. Split H into a development set I and an evaluation set J.

5. Repeat until out of debugging ideas or out of time:

- (a) Perform cross validation on I to select the best matcher. Let this matcher be X.

- (b) Debug X using I. This may change the matcher X, the data, labels, and the set of features, thus changing I and J.

6. Let Y be the best matcher obtained in Step 5. Train Y on I, then apply to J and report the matching accuracy on J.

**Figure 3: The top-level steps of the guide for the EM scenario of matching using supervised learning.**

in general) have been missing, seriously hampering user efforts.

In the rest of this paper we will focus on the development stage, leaving the production stage for subsequent papers.

## **4. HOW-TO GUIDES AND TOOLS**

We now discuss developing how-to guides as well as tools to support these guides. Our goal is twofold:

- First, we show that even for relatively simple EM scenarios (e.g., matching using supervised learning), a good guide can already be quite complex. Thus developing how-to guides is a major challenge, but such guides are absolutely critical in order to successfully guide the user through the EM process.

- Second, we show that each step of the guide, including those that prior work may have viewed as trivial or engineering (e.g., sampling, labeling), can raise many interesting research challenges. We provide preliminary solutions to several such challenges in this paper. But much more remains to be done.

Recall that Magellan currently targets three EM scenarios: matching two tables _A_ and _B_ using (1) supervised learning, (2) rules, and (3) both learning and rules. For space reasons, we will focus on Scenario 1, briefly discussing Scenarios 2- 3 in Section 4.7. For Scenario 1, we further focus on the development stage.

**The Current Guide for Learning-Based EM:** Figure 3 shows the current guide for Scenario 1: matching using supervised learning. The figure lists only the top six steps. While each step may sound like fairly informal advice (e.g., “create a set of features”), the full guide itself (available with Magellan 0.1) is considerably more complex and actually spells out in detail what to do (e.g., run a Magellan command to automatically create the features). We developed this guide based on observing how real-world users (e.g., at WalmartLabs and Johnson Control) as well as students in several UW-Madison classes handled this scenario.

The guide states that to match two tables _A_ and _B_ , the user should load the tables into Magellan (Step 1), do blocking (Step 2), label a sample of tuple pairs (Step 3), use

10

the sample to iteratively find and debug a learning-based matcher (Steps 4-5), then return this matcher and its estimated matching accuracy (Step 6).

We now discuss these steps, possible tools to support them, and tools that we have actually developed. Our goal is to automate each step as much as possible, and where it is not possible, then to provide detailed guidance to the user. We focus on discussing problems with current solutions, the design alternatives, and opportunities for automation. For ease of exposition, we will assume that tables _A_ and _B_ share the same schema.

## **4.1 Loading and Downsampling Tables**

**Downsampling Tables:** We begin by loading the two tables _A_ and _B_ into memory. If these tables are large (e.g., each having 100K+ tuples), we should sample smaller tables _A_<sup>_′_</sup> and _B_<sup>_′_</sup> from _A_ and _B_ respectively, then do the development stage with these smaller tables. Since this stage is iterative by nature, working with large tables can be very time consuming and frustrating to the user.

Random sampling however does not work, because tables _A_<sup>_′_</sup> and _B_<sup>_′_</sup> may end up sharing very few matches, i.e., matching tuples (especially if the number of matches between _A_ and _B_ is small to begin with). Thus we need a tool that samples more intelligently, to ensure a reasonable number of matches between _A_<sup>_′_</sup> and _B_<sup>_′_</sup> .

We have developed such a tool, shown as the Magellan command _c_ 1 in Figure 4. This command first randomly selects _B_ _~~s~~ ize_ tuples from table _B_ to be table _B_<sup>_′_</sup> . For each tuple _x ∈ B_<sup>_′_</sup> , it finds a set _P_ of _k/_ 2 tuples from _A_ that may match _x_ (using the heuristic that if a tuple in _A_ shares many tokens with _x_ , then it is more likely to match _x_ ), and a set _Q_ of _k/_ 2 tuples randomly selected from _A \ P_ . Table _A_<sup>_′_</sup> will consist of all tuples in such _P_ s and _Q_ s. The idea is for _A_<sup>_′_</sup> and _B_<sup>_′_</sup> to share some matches yet be as representative of _A_ and _B_ as possible.

To find _P_ , the command relies on the heuristic that if two tuples share many tokens, then they are likely to match. Thus, it builds an inverted index _I_ of ( _token, tuple_ _~~i~~ d_ ) over table _A_ , probes _I_ to find all tuples in _A_ that share tokens with _x_ , rank these tuples in decreasing number of shared tokens, then take (up to) the top _k/_ 2 tuples to be the set _P_ . Note that index _I_ is built only once, at the start of the command. The command then randomly samples _k −|P |_ tuples in _A \ P_ to be the set _Q_ .

**More Sophisticated Downsampling Solutions:** The above command was fast and quite effective in our experiments. However it has a limitation: it may not get all important matching categories into _A_<sup>_′_</sup> and _B_<sup>_′_</sup> . If so, the EM workflow created using _A_<sup>_′_</sup> and _B_<sup>_′_</sup> may not work well on the original tables _A_ and _B_ .

For example, consider matching companies. Tables _A_ and _B_ may contain two matching categories: (1) tuples with similar company names and addresses match because they refer to the same company, and (2) tuples with similar company names but different addresses may still match because they refer to different branches of the same company. Using the above command, tables _A_<sup>_′_</sup> and _B_<sup>_′_</sup> may contain many tuple pairs of Case 1, but no or very few pairs of Case 2.

To address this problem, we are working on a better “downsampler”. Our idea is to use clustering to create groups of matching tuples, then analyze these groups to infer match-

c1: down_sample_tables (A, B, B_size, k) c2: debug_blocker (A, B, C, output_size = 200) c3: get_features_for_matching (A, B)

c4: select_matcher (matchers, table, exclude_attrs, target_attr, k = 5) c5: vis_debug_dt (matcher, train, test, exclude_attrs, target_attr)

**Figure 4: Sample commands discussed in Section 4. Magellan has 53 such commands.**

**Figure 5: Magellan console in interactive IPython.**

ing categories, then sample from the categories. Major challenges here include how to effectively cluster tuples from the large tables _A_ and _B_ , and how to define and infer matching categories accurately.

## **4.2 Blocking to Create Candidate Tuple Pairs**

In the next step, we apply blocking to the two tables _A_<sup>_′_</sup> and _B_<sup>_′_</sup> to remove obviously non-matched tuple pairs. Ideally, this step should be automated (as much as possible). Toward this goal, we distinguish three cases.

(1) We already know which matcher we want to use. Then it may be possible to analyze the matcher to infer a blocker, thereby completely automating the blocking step. For example, when matching two sets of strings (a special case of EM [16]), often we already know the matcher we want to use (e.g., _jaccard_ ( _x, y_ ) _>_ 0 _._ 8, i.e., predicting two strings _x_ and _y_ matched if their Jaccard score exceeds 0.8). Prior work [16] has analyzed such matchers to infer efficient blockers that do not remove true matches. Thus, debugging the blocker is also not necessary.

(2) We do not know yet which matcher we want to use, but we have a set _T_ of tuple pairs labeled matched / no-matched. Then it may be possible to partially automate the blocking step. Specifically, the system can use _T_ to learn a blocker and propose it to the user (e.g., training a random forest

11

<!-- Start of picture text -->
Table A Table B<br>Id Name Zip code Id Name Zip code A.Id B.Id<br>a1  Bill George 94107 b1  William George 94107 a1  b1<br>a2  Mark Levene 94108 b2  Aaron Miller 94122 block a2  b4  1. (a2, b4)<br>a3  Levent Koc 94132 b3  Levent Koch 94122 attr. equivalence on zip code a4  b2  2. (a3, b3)<br>a4  Michael Franklin 94122 b4  Mark Levene 94107 a4  b3<br>(a) (b) (c)<br><!-- End of picture text -->

**Figure 6: An example for debugging blocker output.**

**Figure 7: The GUI of the blocking debugger.**

then extracting the negative rules of the forest as blocker candidates [26]). The user still has to debug the blocker to check that it does not accidentally remove too many true matches.

(3) We do not know yet which matcher we want to use, and we have no labeled data. This is the case considered in this paper, since all we have so far are the two tables _A_<sup>_′_</sup> and _B_<sup>_′_</sup> . In this case the user often faces three problems (which have not been addressed by current work): (a) how to select the best blocker, (b) how to debug a given blocker, and (c) how to know when to stop? Among these, the first problem is open to partial automation.

**Selecting the Best Blocker:** A straightforward solution is to label a set of tuple pairs (e.g., selected using active learning [26]), then use it to automatically propose a blocker, as in Case 2. To propose good blockers, however, this solution may require labeling hundreds of tuple pairs [26], incurring a sizable burden on the user.

This solution may also be unnecessarily complex. In practice, a user often can use domain knowledge to quickly propose good blockers, e.g., “matching books must share the same ISBN”, in a matter of minutes. Hence, our howto guide tries to help the user identify these “low-hanging fruits” first.

Specifically, many blocking solutions have been developed, e.g., overlap, attribute equivalence (AE), sorted neighborhood (SNB), hash-based, rule-based, etc. [16]. From our experience, we recommend that the user try successively more complex blockers, and stop when the number of the tuple pairs surviving blocking is already sufficiently small. Specifically, the user can try overlap blocking first (e.g., “matching tuples must share at least _k_ tokens in an attribute _x_ ”), then AE (e.g., “matching tuples must share the same value for an attribute _y_ ”). These blockers are very fast, and can significantly cut down on the number of candidate tuple pairs. Next, the user can try other well-known blocking methods (e.g., SNB, hash) if appropriate. This means the user can use multiple blockers and combine them in a flexible fashion (e.g., applying AE to the output of overlap blocking).

Example 2. _Figure 5 shows a case where the user has loaded two tables A and B into Python, inspected the tables by using the visualization capabilities of the pandas Python package, then performed AE blocking on_ zipcode _(see the line starting with_ In [6] _)._

Finally, if the user still wants to reduce the number of candidate tuple pairs further, then he or she can try rule-based blocking. It is difficult to manually come up with good blocking rules. So we will develop a tool to automatically propose rules, as in Case 2, using the technique in [26], which uses active learning to select tuple pairs for the user to label.

**Debugging Blockers:** Given a blocker _L_ , how do we know if it does not remove too many matches? We have developed a debugger to answer this question, shown as command _c_ 2 in Figure 4. Suppose applying _L_ to _A_<sup>_′_</sup> and _B_<sup>_′_</sup> produces a set _C_ of tuple pairs ( _a ∈ A_<sup>_′_</sup> _, b ∈ B_<sup>_′_</sup> ). Then _D_ = _A_<sup>_′_</sup> _× B_<sup>_′_</sup> _\ C_ is the set of all tuple pairs removed by _L_ . The debugger examines _D_ to return a list of _k_ tuple pairs in _D_ that are most likely to match ( _k_ = 200 is the default). The user _U_ examines this list. If _U_ finds many matches in the list, then that means blocker _L_ has removed too many matches. _U_ would need to modify _L_ to be less “aggressive”, then apply the debugger again. Eventually if _U_ finds no or very few matches in the list, _U_ can assume that _L_ has removed no or very few matches, and thus is good enough.

Example 3. _Given the two tables A and B in Figure 6.a, attribute equivalence-based blocking on_ zipcode _will produce the set of tuple pairs in Figure 6.b. Applying the debugger to Tables A and B and the set of tuple pairs (that survive blocking) may produce the ranked list of two tuple pairs in Figure 6.c. (Figure 7 shows a screen shot of how the ranked list is typically presented to the user in_ Magellan _.)_

_When the user examines these two tuple pairs, he/she may realize that both of them are likely to be matches. This means that the blocker has been too aggressive, in that it has dropped too many true matches. In this case, the user may decide not to use this attribute equivalance-based blocker._

Developing the above debugger raises two challenges. First, how can it judge that a tuple pair is likely to match? Second, how can it search _D_ very fast (given that debugging is interactive by nature)? To address the first challenge, we first select a set of attributes judged to be discriminative, in that if two tuples ( _a ∈ A_<sup>_′_</sup> _, b ∈ B_<sup>_′_</sup> ) share similar or identical values for most of these attributes, then they are likely to match. Let _x_ be an attribute, we compute

- _unique_ ( _x, A_<sup>_′_</sup> ) to be the number of unique values of _x_ in _A_<sup>_′_</sup> divided by the number of non-empty values of _x_ in _A_<sup>_′_</sup> ,

12

- _missing_ ( _x, A_<sup>_′_</sup> ) to be the number of missing values of _x_ in _A_<sup>_′_</sup> divided by the number of tuples in _A_<sup>_′_</sup> , and

- _s_ ( _x, A_<sup>_′_</sup> ) = _unique_ ( _x, A_<sup>_′_</sup> ) + 1 _− missing_ ( _x, A_<sup>_′_</sup> ).

The score _s_ ( _x, A_<sup>_′_</sup> ) indicates how discriminative attribute _x_ is in table _A_<sup>_′_</sup> . Intuitively, the higher _unique_ ( _x, A_<sup>_′_</sup> ), the more likely that a value of _x_ can uniquely identify a tuple in _A_<sup>_′_</sup> , unless _x_ has a lot of missing values, which is taken into account using 1 _− missing_ ( _x, A_<sup>_′_</sup> ).

Defining _s_ ( _x, B_<sup>_′_</sup> ) similarly, we can define a discriminativeness score for _x_ across both tables: _s_ ( _x_ ) = _s_ ( _x, A_<sup>_′_</sup> ) _· s_ ( _x, B_<sup>_′_</sup> ). We then select the top _k_ attributes with the highest _s_ ( _x_ ) scores (where _k_ is pre-specified), to be used in the debugger.

Let the set of selected attributes be _T_ . For each tuple _a ∈ A_<sup>_′_</sup> , let _t_ ( _a_ ) be the string resulting from concatenating the values of the selected attributes. Define _t_ ( _b_ ) similarly for each tuple _b ∈ B_<sup>_′_</sup> . Let _J_ ( _t_ ( _a_ ) _, t_ ( _b_ )) be the Jaccard score between _t_ ( _a_ ) and _t_ ( _b_ ), assuming each of these strings have been tokenized into a set of 3-grams. Then the debugger returns the top _k_ tuple pairs ( _a, b_ ) in _D_ = _A_<sup>_′_</sup> _× B_<sup>_′_</sup> _\ C_ with the highest _J_ ( _t_ ( _a_ ) _, t_ ( _b_ )) scores. Intuitively, the debugger states that these pairs are likely to be matches, so the user should check them. To find these pairs fast, the debugger uses indexes on the tables. We omit further details for space reasons.

**Knowing When to Stop Modifying the Blockers:** How do we know when to stop tuning a blocker _L_ ? Suppose applying _L_ to _A_<sup>_′_</sup> and _B_<sup>_′_</sup> produces the set of tuple pairs _block_ ( _L, A_<sup>_′_</sup> _, B_<sup>_′_</sup> ). The conventional wisdom is to stop when _block_ ( _L, A_<sup>_′_</sup> _, B_<sup>_′_</sup> ) fits into memory or is already small enough so that the matching step can process it efficiently.

In practice, however, this often does not work. For example, since we work with _A_<sup>_′_</sup> and _B_<sup>_′_</sup> , _samples_ from the original tables, monitoring _|block_ ( _L, A_<sup>_′_</sup> _, B_<sup>_′_</sup> ) _|_ does not make sense. Instead, we want to monitor _|block_ ( _L, A, B_ ) _|_ . But applying _L_ to the large tables _A_ and _B_ can be very time consuming, making the iterative process of tuning _L_ impractical. Further, in many practical scenarios (e.g., e-commerce), the data to be matched can arrive in batches, over weeks, rendering moot the question of estimating _|block_ ( _L, A, B_ ) _|_ .

As a result, in many practical settings users want blockers that have (1) high pruning power, i.e., maximizing 1 _− |block_ ( _L, A_<sup>_′_</sup> _, B_<sup>_′_</sup> ) _|/|A_<sup>_′_</sup> _× B_<sup>_′_</sup> _|_ , and (2) high recall, i.e., maximizing the ratio of the number of matches in _block_ ( _L, A_<sup>_′_</sup> _, B_<sup>_′_</sup> ) divided by the number of matches in _A_<sup>_′_</sup> _× B_<sup>_′_</sup> .

Users can measure the pruning power, but so far they have had no way to estimate recall. This is where our debugger comes in. In our experiments (see Section 6) users reported they had used our debugger to find matches that the blocker _L_ had removed, and when they found no or only a few matches, they concluded that _L_ had achieved high recall and stopped tuning the blocker.

## **4.3 Sampling and Labeling Tuple Pairs**

Let _L_ be the blocker we have created. Suppose applying _L_ to tables _A_<sup>_′_</sup> and _B_<sup>_′_</sup> produces a set of tuple pairs _C_ . In the next step, user _U_ should take a sample _S_ from _C_ , then label the pairs in _S_ as matched / no-matched, to be used later for training matchers, among others.

At a first glance, this step seems very simple: why not just take a random sample and label it? Unfortunately in practice this is far more complicated.

**Figure 8: The GUI of the matching debugger.**

For example, suppose _C_ contains relatively few matches (either because there are few matches between _A_<sup>_′_</sup> and _B_<sup>_′_</sup> , or because blocking was too liberal, resulting in a large _C_ ). Then a random sample _S_ from _C_ may contain no or few matches. But the user _U_ often does not recognize this until _U_ has labeled most of the pairs in _S_ . This is a waste of _U_ ’s time and can be quite serious in cases where labeling is time consuming or requires expensive domain experts (e.g., labeling drug pairs when we worked with Marshfield Clinic). Taking another random sample does not solve the problem because it is likely to also contain no or few matches.

To address this problem, our guide builds on [26] to propose that user _U_ sample and label in iterations. Specifically, suppose _U_ wants a sample _S_ of size _n_ . In the first iteration, _U_ takes and labels a random sample _S_ 1 of size _k_ from _C_ , where _k_ is a small number. If there are enough matches in _S_ 1, then _U_ can conclude that the “density” of matches in _C_ is high, and just randomly sample _n − k_ more pairs from _C_ .

Otherwise, the “density” of matches in _C_ is low. So _U_ must re-do the blocking step, perhaps by creating new blocking rules that remove more non-matching tuple pairs in _C_ , thereby increasing the density of matches in _C_ . After blocking, _U_ can take another random sample _S_ 2 also of size _k_ from _C_ , then label _S_ 2. If there are enough matches in _S_ 2, then _U_ can conclude that the density of matches in _C_ has become high, and just randomly sample _n −_ 2 _k_ more pairs from _C_ , and so on.

## **4.4 Selecting a Matcher**

Once user _U_ has labeled a sample _S_ , _U_ uses _S_ to select a good initial learning-based matcher. Today most EM systems supply the user with a set of such matchers, e.g., decision tree, Naive Bayes, SVM, etc., but do not tell the user how to select a good one.

Our guide addresses this problem. Specifically, user _U_ first calls the command _c_ 3 in Figure 4 to create a set of features _F_ = _{f_ 1 _, . . . , fm}_ , where each feature _fi_ is a function that maps a tuple pair ( _a, b_ ) into a value. This command creates all possible features between the attributes of tables _A_<sup>_′_</sup> and _B_<sup>_′_</sup> , using a set of heuristics. For example, if attribute _name_ is textual, then the command creates feature _name_ ~~3~~ _gram_ _~~j~~ ac_ that returns the Jaccard score between the 3-gram sets of the two names (of tuples _a_ and _b_ ). Next, _U_ converts each tuple pair in the labeled set _S_ into a feature vector (using features in _F_ ), thus converting _S_ into a set _H_ of feature vectors. Next, _U_ splits _H_ into a development set _I_ and an evaluation set _J_ .

Let _M_ be the set of all learning-based matchers supplied by the EM system. Next, _U_ uses command _c_ 4 in Figure 4

13

to perform cross validation on _I_ for all matchers in _M_ , then examines the results to select a good matcher. Command _c_ 4 highlights the matcher with the highest accuracy. However, if a matcher achieves just slightly lower accuracy (than the best one) but produces results that are easier to explain and debug (e.g., a decision tree), then _c_ 4 highlights that matcher as well, for the user’s consideration.

Thus, the entire process of selecting a matcher can be automated (if the user does not want to be involved), and in fact Magellan does provide a single command to execute the entire process.

## **4.5 Debugging a Matcher**

Let the selected matcher be _X_ . In the next step user _U_ debugs _X_ to improve its accuracy. Such debugging is critical in practice, yet has received very little attention in the research community.

Our guide suggests that user _U_ debug in three steps: (1) identify and understand the matching mistakes made by _X_ , (2) categorize these mistakes, and (3) take actions to fix common categories of mistakes.

**Identifying and Understanding Matching Mistakes:** _U_ should split the development set _I_ into two sets _P_ and _Q_ , train _X_ on _P_ then apply it to _Q_ . Since _U_ knows the labels of the pairs in _Q_ , he or she knows the matching mistakes made by _X_ in _Q_ . These are _false positives_ (non-matching pairs predicted matching) and _false negatives_ (matching pairs predicted not). Addressing them helps improve precision and recall, respectively.

Next _U_ should try to understand why _X_ makes each mistake. For example, let ( _a, b_ ) _∈ Q_ be a pair labeled “matched” for which _X_ has predicted “not matched”. To understand why, _U_ can start by using a debugger that explains how _X_ comes to that prediction. For example, if _X_ is a decision tree then the debugger (invoked using command _c_ 5 in Figure 4) can show the path from the root of the tree to the leaf that ( _a, b_ ) has traversed. Examining this path, as well as the pair ( _a, b_ ) and its label, can reveal where things go wrong. In general things can go wrong in four ways:

- The data can be dirty, e.g., the price value is incorrect.

- The label can be wrong, e.g., ( _a, b_ ) should have been labeled “not matched”.

- The feature set is problematic. A feature is misleading, or a new feature is desired, e.g., we need a new feature that extracts and compares the publishers.

- The learning algorithm employed by _X_ is problematic, e.g., a parameter such as “maximal depth to be searched” is set to be too small.

Currently Magellan has debuggers for a set of learning-based matchers, e.g., decision tree, random forest (Figure 8 shows a screen shot of the matching debugger for one of these matcher types.) We are working on improving these debuggers and developing debuggers for more learning algorithms.

**Categorizing Matching Mistakes:** After _U_ has examined all or a large number of matching mistakes, he or she can categorize them, based on problems with data, label, feature, and the learning algorithm.

Examining all or most mistakes is very time consuming. Thus a consistent feedback we have received from real-world

<!-- Start of picture text -->
clean,<br>A<br>extract, transform  Candidate<br>block Set C  match<br>clean,<br>B<br>extract, transform<br><!-- End of picture text -->

**Figure 9: The EM workflow for the learning-based matching scenario.**

users is that they would love a tool that can automatically examine and give a preliminary categorization of the types of the matching mistakes. As far as we can tell, no such tool exists today.

**Handling Common Categories of Mistakes:** Next _U_ should try to fix common categories of mistakes by modifying the data, labels, set of features, and the learning algorithm. This part often involves data cleaning and extraction (IE), e.g., normalizing all values of attribute “affiliation”, or extracting publishers from attribute “desc” then creating a new feature comparing the publishers.

This part is often also very time consuming. Real-world users have consistently indicated needing support in at least two areas. First, they want to know exactly what kinds of data cleaning and IE operations they need to do to fix the mistakes. Naturally they want to do as minimally as possible. Second, re-executing the entire EM process after each tiny change to see if it “fixes” the mistakes is very time consuming. Hence, users want an “what-if” tool that can quickly show the effect of a hypothetical change.

**Proxy Debugging:** Suppose we need to debug a matcher _X_ but there is no debugger for _X_ , or the debugger exists but is not very informative. In this case _X_ is effectively a “blackbox”. To address this problem, in Magellan we have introduced a novel debugging method. In particular, we propose to train another matcher _X_<sup>_′_</sup> for which there is a debugger, then use that debugger to debug _X_<sup>_′_</sup> , instead of _X_ . This “proxy debugging” process cannot fix problems with the learning algorithm of _X_ , but it can reveal problems with the data, labels, features, and fixing them can potentially improve the accuracy of _X_ itself. Section 6.2 shows cases of proxy debugging working quite well in practice.

**Selecting a Matcher Again:** So far we have discussed selecting a good initial learning-based matcher _X_ , then debugging _X_ using the development set _I_ . To debug, user _U_ splits _I_ into training set _P_ and testing set _Q_ , then identifies and fixes mistakes in _Q_ . Note that this splitting of _I_ into _P_ and _Q_ can be done multiple times. Subsequently, since the data, labels, and features may have changed, _U_ would want to do cross validation again to select a new “best matcher”, and so on (see Step 5 in Figure 3).

## **4.6 The Resulting EM Workflow**

After executing the above steps, user _U_ has in effect created an EM workflow, as shown in Figure 9. Since this workflow will be used in the production stage, it takes as input the two original tables _A_ and _B_ . Next, it performs a set of data cleaning, IE, and transformation operations on these tables. These operations are derived from the debugging step discussed in Section 4.5.

14

Next, the workflow applies the blockers created in Section 4.2 to obtain a set of candidate tuple pairs _C_ . Finally, the workflow applies the learning-based matcher created in Section 4.5 to the pairs in _C_ .

Note that the steps of sampling and labeling a sample _S_ do not appear in this workflow, because we need them only in the development stage, in order to create, debug, and train matchers. Once we have found a good learning-based matcher (and have trained it using _S_ ), we do not have to execute those steps again in the production stage.

## **4.7 How-to Guides for Scenarios with Rules**

Recall that Magellan currently targets three EM scenarios. So far we have discussed a how-to guide and tools for Scenario 1: matching using supervised learning. We now briefly discuss Scenarios 2 and 3.

Scenario 2 uses only rules to match. This is desirable in practice for various reasons (e.g., when matching medicine it is often important that we can explain the matching decision). For this scenario, we have developed guides and tools to help users (a) create matching rules manually, (b) create rules using a set of labeled tuple pairs, or (c) create rules using active learning.

Scenario 3 uses both supervised learning and rules. Users often want this when using neither learning nor rules alone gives them the desired accuracy. For this scenario, we have also developed a guide and tools to help users. Our guide suggests that users do learning-based EM first, as described earlier for Scenario 1, then add matching rules “on top” of the learning-based matcher, to improve matching accuracy. We omit further details for space reasons.

## **5. DESIGNING FOR AN OPEN WORLD**

So far we have discussed how-to guides and tools to support the guides. We now turn to the challenge of designing these tools as commands in Python.

This challenge turned out to be highly non-trivial, as we will see. It raises a fundamental question: what do we mean by “building on top of a data analysis stack”? To answer, we introduce the notion of closed-world vs. open-world systems for EM contexts. We show that Magellan should be built as an open-world system, but building such systems raises difficult problems such as designing appropriate data structures and managing metadata. Finally, we discuss how Magellan addresses these problems.

## **5.1 Closed-World vs. Open-World Systems**

A closed-world system controls its own data. This data can only be manipulated by its own commands. For this system, its own world is the only world. There is nothing else out there and thus it does not have a notion of having to “play well” with other systems. It is often said that RDBMSs are such closed-world systems. Virtually all current EM systems can also be viewed as closed-world systems.

In contrast, an open-world system _K_ is aware that there is a whole world “out there”, teeming with other systems, and that it will have to interact with them. The system therefore possesses the following characteristics:

- _K_ expects other systems to be able to manipulate _K_ ’s own data.

- _K_ may also be called upon by other systems to manipulate their own data.

_• K_ is designed in a way that facilitates such interaction.

Thus, by building Magellan on the Python data analysis stack we mean building an open-world system as described above (where “other systems” are current and future Python packages in the stack). This is necessary because, as discussed earlier, in order to do successful EM, Magellan will need to rely on a wide range of external systems to supply tools in learning, mining, visualization, cleaning, IE, etc. Building an open-world system however raises difficult problems. In what follows we discuss problems with data structures and metadata. (We have also encountered several other problems, such as missing values, data type mismatch, package version incompatabilities, etc., but will not discuss them in this paper.)

## **5.2 Designing Data Structures**

At the heart of Magellan is a set of tables. The tuples to be matched are stored in two tables _A_ and _B_ . The intermediate and final results can also be stored in tables. Thus, an important question is how to implement the tables. A popular Python package called pandas has been developed to store and process tables, using a data structure called “data frame”. Thus, the simplest solution is to implement Magellan’s tables as data frames. A problem is that data frames cannot store metadata, e.g., a constraint that an attribute is a key of a table.

A second choice is to define a new Python class called MTable, say, where each MTable object has multiple fields, one field points to a data frame holding the tuples, another field points to the key attributes, and so on.

Yet a third choice is to subclass the data frame class to define a new Python class called MDataFrame, say, which have fields such as “keys”, “creation-date”, etc. besides the inherited data frame holding the tuples.

From the perspective of building open-world systems, as discussed in Section 5.1, the last two choices are bad because they make it difficult for external systems to operate on Magellan’s data. Specifically, MTable is a completely unfamiliar class to existing Python packages. So commands in these packages cannot operate on MTable objects directly. We would need to redefine these commands, a time-consuming and brittle process.

MDataFrame is somewhat better. Since it is a subclass of data frame, any existing command (external to Magellan) that knows data frames can operate on MDataFrame objects. Unfortunately the commands may return inappropriate types of objects. For example, a command deleting a row in an MDataFrame object would return a data frame object, because being an external command it is not aware of the MDataFrame class. This can be quite confusing to users, who want external commands to work smoothly on Magellan’s objects.

For these reasons, we take the first choice: storing Magellan’s tables as data frames. Since virtually any existing Python package that manipulates tables can manipulate data frames, this maximizes the chance that commands from these packages can work seamlessly on Magellan’s tables.

In general, we propose that an open-world system _K_ use the data structures that are most common to other systems to store its data. This brings two important benefits: it is easier for other systems to operate on _K_ ’s data, and there will be far more tools available to help _K_ manipulate its own data. If it is not possible to use common data structures,

15

_K_ should provide procedures that convert between its own data structures and the ones commonly used by other openworld systems.

## **5.3 Managing Metadata**

We have discussed storing Magellan’s tables as data frames. Data frames however cannot hold metadata (e.g., key and foreign key constraints, date last modified, ownership). Thus we will store such metadata in a central catalog.

Regardless of where we store the metadata, however, letting external commands directly manipulate Magellan’s data leads to a problem: the metadata can become inconsistent. For example, suppose we have created a table _A_ and stored in the central catalog that “sid” is a key for _A_ . There is nothing to prevent a user _U_ from invoking an external command (of a non-Magellan package) on _A_ to remove “sid”. This command however is not aware of the central catalog (which is internal to Magellan). So after its execution, the catalog still claims that “sid” is a key for _A_ , even though _A_ no longer contains “sid”. As another example, an external command may delete a tuple from a table participating in a key-foreign key relationship, rendering this relationship invalid, while the catalog still claims that it is valid.

In principle we can rewrite the external commands to be metadata aware. But given the large number of external commands that Magellan users may want to use, and the rapid changes for these commands, rewriting all or most of them in one shot is impractical. In particular, if a user _U_ discovers a new package that he or she wants to use, we do not want to force _U_ to wait until Magellan’s developers have had a chance to rewrite the commands in the package to be metadata aware. But allowing _U_ to use the commands immediately, “as is”, can lead to inconsistent metadata, as discussed above.

To address this problem, we design each Magellan’s command _c_ from the scratch to be metadata aware. Specifically, we write _c_ such that at the start, it checks for all constraints that it requires to be true, in order for it to function properly. For example, _c_ may know that in order to operate on table _A_ , it needs a key attribute. So it looks up the central catalog to obtain the constraint that “sid” is a key for _A_ . Command _c_ then checks this constraint to the extent possible. If it finds this constraint invalid, then it alerts the user and asks him or her to fix this constraint.

Command _c_ will not proceed until all required constraints have been verified. During its execution, it will try to manage metadata properly. In addition, if it encounters an invalid constraint it will alert the user, but will continue its execution, as this constraint is not critical for its correct execution (those constraints have been checked at the start of the command). For example, if it finds a dangling tuple due to a violation of a foreign key constraint, it may just alert the user, ignore the tuple, and then continue.

## **6. EMPIRICAL EVALUATION**

We now empirically evaluate Magellan. It is difficult to evaluate such a system in large-scale experiments with realworld data and users. To address this challenge, we evaluated Magellan in two ways. First, we asked 44 UW-Madison students to apply Magellan to many real-world EM scenarios on the Web. Second, we provided Magellan to real users at several organizations (WalmartLabs, Johnson Control, and

Marshield Clinic) and reported on their experience. We now elaborate on these two sets of experiments.

## **6.1 Large-Scale Experiments on Web Data**

Our largest experiment was with 24 teams of CS students (a total of 44 students) at UW-Madison in a Fall 2015 data science class. These students can be considered the equivalents of power users at organizations. They know Python but are not experts in EM.

We asked each team to find two data-rich Web sites, extract and convert data from them into two relational tables, then apply Magellan to match tuples across the tables. The first four columns of Table 4 show the teams, domains, and the sizes of the two tables, respectively. Note that two teams may cover the same domain, e.g., “Movies”, but extract from different sites. Overall, there are 12 domains, and the tables have 7,313 tuples on average, with 5-17 attributes.

We asked each team to do the EM scenario of supervised learning followed by rules, and aim for precision of at least 90% with recall as high as possible. This is a very common scenario in practice.

**The Baseline Performance:** The columns under “Initial Learning-Based Matcher (A)” show the matching accuracies achieved by the best learning-based matcher (after cross validation, see Section 4.4): _P_ = 56 _−_ 100% _, R_ = 37 _._ 5 _−_ 100% _, F_ 1 = 56 _−_ 99 _._ 5%. These results show that many of these tables are not easy to match, as the best learning-based matcher selected after cross validation does not achieve high accuracy. In what follows we will see how Magellan was able to significantly improve these accuracies.

**Using the How-to Guide:** The columns under “Final Learning+Rule Matcher (D)” show the final matching accuracies that the teams obtained: _P_ = 91 _._ 3 _−_ 100% _, R_ = 64 _._ 7 _−_ 100% _, F_ 1 = 78 _._ 6 _−_ 100%. All 24 teams achieved precision exceeding 90%, and 20 teams also achieved recall exceeding 90%. (Four teams had recall below 90% because their data were quite dirty, with many missing values.) All teams reported being able to follow the how-to guide. Together with qualitative feedback from the teams, this suggests that users can follow Magellan’s how-to guide to achieve high matching accuracy on diverse data sets. We elaborate on these results below, broken down by blocking and matching.

**Blocking and Debugging Blockers:** All teams used 1- 5 blockers (e.g., attribute equivalence, overlap, rule-based), for an average of 3. On average 3 different types of blockers were used per team. This suggests that it is relatively easy to create a blocking pipeline with diverse blocker types.

All teams debugged their blockers, in 1-10 iterations, for an average of 5. 18 out of 24 teams used our debugger (see Section 4.2), and reported that it helped in four ways.

**_(a) Cleaning data:_** By examining tuple pairs (returned by the debugger) that are matches accidentally removed by blocking, 12 teams discovered data that should be cleaned. For example, one team removed the edition information from book titles, and another team normalized the date formats in the input tables.

**_(b) Finding the correct blocker types and attributes:_** 12 teams were able to use the debugger for these purposes. For example, one team found that using attribute equivalence (AE) blocker over “phone” removed many matches,

16

|Team|Domain|Size of<br>Tbl A|Size of<br>Tbl B|Cand.<br>Set|Initial<br>M|Learning<br>atcher(A|-Based<br>)|Final<br>|Learning-<br>Matcher(B|Based<br>)|Num. of<br>Iterations|Fina<br>Rules|l Learni<br>Match|ng +<br>er(D)|Num. of<br>Iterations|Diff. in F1<br>between<br>D d A|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||ae|ae|Size|P|R|F1|P|R|F1|(C)|P|R|F1|(E)|() an ()<br>in %|
|1|Vehicles|4786|9003|8009|71.2|71.2|71.2|91.43|94.12|92.75|4|100|100|100|2|30.27|
|2|Movies|7391|6408|78079|99.28|95.13|97.04|98.21|100|99.1|2|100|100|100|1|2.12|
|3|Movies|3000|3000|1000000|98.9|99.44|99.5|98.63|98.63|98.63|1|98.63|98.63|98.63|0|-0.87|
|4|Movies|3000|3000|36000|68.2|69.16|68.6|98|100|98.99|3|98|100|98.99|1|44.3|
|5|Movies|6225|6392|54028|100|95.23|97.44|100|100|100|3|100|100|100|1|2.63|
|6|Restaurants|6960|3897|10630|100|37.5|54.55|100|88.89|94.12|3|100|88.89|94.12|1|72.54|
|7|Electronic Products|4559|5001|823832|73|51|59|73.3|64.71|68.75|2|100|64.71|78.57|1|33.17|
|8|Music|6907|55923|58692|92|79.31|85.19|90.48|82.61|86.36|2|100|92.16|95.92|2|1.37|
|9|Restaurants|9947|28787|400000|100|78.5|87.6|94.44|97.14|95.77|4|94.44|97.14|95.77|0|9.33|
|10|Cosmetic|11026|6445|36026|56|56|56|96.67|87.88|92.06|3|96.43|<br>87.1|91.53|4|64.39|
|11|E-Books|6482|14110|13652|96.67|96.67|96.67|100|95.65|97.78|4|100|98.33|99.13|1|1.15|
|12|Beer|4346|3000|4334961|84.5|59.6|65.7|100|60.87|75.68|4|91.3|91.3|91.3|4|15.19|
|13|Books|3506|3508|2016|93.46|100|96.67|91.6|100|95.65|2|91.6|100|95.65|0|-1.06|
|14|Books|3967|3701|4029|74.17|82.2|82.5|100|84.85|91.8|3|100|84.85|91.8|5|11.27|
|15|Anime|4000|4000|138344|95.9|88.9|92.2|100|100|100|2|100|100|100|1|8.46|
|16|Books|3021|3098|931|74.2|100|85.2|96.34|84.95|90.29|2|94.51|92.47|93.48|1|5.97|
|17|Movies|3556|6913|504|94.2|99.33|96.6|95.04|94.26|94.65|2|95.04|94.26|94.65|1|-2.02|
|18|Books|8600|9000|492|91.6|100|84.8|94.8|100|90.2|3|100|92.31|96|1|6.37|
|19|Restaurants|11840|5223|5278|98.6|93.8|96.1|95.6|94.02|95.57|2|100|94.12|96.97|1|-0.55|
|20|Books|3000|3000|257183|94.24|72.88|81.71|90.91|83.33|86.96|2|92.31|<br>100|96|1|6.43|
|21|Literature|3885|3123|1590633|84.4|86.9|85.5|100|95.65|97.83|3|100|95.65|97.83|0|14.42|
|22|Restaurants|3014|5883|78190|100|93.59|96.55|100|100|100|5|100|100|100|0|3.57|
|23|E-Books|6501|14110|18381|94.6|92.5|93.4|94.6|97.22|95.89|2|100|100|100|1|2.67|
|24|BabyProducts|10000|5000|11000|78.6|44.8|57.7|96.43|72.97|83.08|5|100|72.97|84.37|2|43.99|

**Table 4: Large-scale experiments with Magellan on Web data.**

because the phone numbers were not updated. So they decided to use “zipcode” instead. Another team started with AE over “name” then realized that the blocker did not work well because many names were misspelled. So they decided to use a rule-based blocker instead.

**_(c) Tuning blocker parameters:_** 18 teams used the debugger for this purpose, e.g., to change the overlap size for “address” in an overlap blocker, or to use a different threshold for a Jaccard measure in a rule-based blocker.

**_(d) Knowing when to stop:_** 12 teams explicitly mentioned in their reports that when the debugger returned no or very few matches, they concluded that the blocking pipeline had done well, and stopped tuning this pipeline.

Teams reported spending 4-32 hours on blocking (including reading documentations). Overall, 21 out of 24 teams were able to prune away more than 95% of _|A × B|_ , with an average reduction of 97.3%, suggesting that they were able to construct blocking with high pruning rate.

Feedback-wise, teams reported liking (a) the ability to create rich and flexible blocking sequences with different types of blockers, (b) the diverse range of blocker types provided by Magellan, and (c) the debugger. They complained that certain types of blockers (e.g., rule-based ones) were still slow (an issue that we are currently addressing).

**Matching and Debugging Matchers:** Recall from Section 4.5 that after cross validation on labeled data to select the best learning-based matcher _X_ , user _U_ iteratively debugged _X_ to improve its accuracy. Teams performed 1-5 debugging iterations, for an average of 3 (see Column “Num

of Iterations (C)” in Table 4). The actions they took were: **_(a) Feature selection:_** 21 teams added and deleted features, e.g., adding more phone related features, removing style related features.

**_(b) Data cleaning:_** 12 teams cleaned data based on the debugging result, e.g., normalizing colors using a dictionary, detecting that the tables have different date formats. 16 teams found and fixed incorrect labels during debugging.

**_(c) Parameter tuning:_** 3 teams tuned the parameters of the learning algorithm, e.g., modifying the maximum depth of decision tree based on debugging results.

These debugging actions helped improve accuracies significantly, from 56-100% to 73.3-100% precision, and 37.5-100% to 61-100% recall (compare columns under “A” with those under “B” in Table 4).

Adding rules further improves accuracy. 19 teams added 1-5 rules, found in 1-5 iterations (see column “E”). This improved precision from 73.3-100% to 91.3-100% and recall from 61-100% to 64.7-100% (compare columns under “D” with those under “B”). Overall, Magellan improved the baseline accuracy in columns “A” significantly, by as much as 72.5% _F_ 1, for an average of 18.8% _F_ 1. For 3 teams, however, accuracy dropped by 0.87-2.02% _F_ 1. This is because the baseline _F_ 1s already exceeded 94%, and when teams tried to add rules to increase _F_ 1 further, they overfit the development set.

Teams reported spending 5-50 hours, for an average of 12 hours (including reading documentation and labeling samples) on matching. They reported liking debugger support, ease of creating custom features for matchers, and support

17

for rules to improve learning-based matching. They would like to have more debugger support, including better ordering and visualization of matching mistakes.

## **6.2 Experience with Organizational Data**

We now describe our experience with Magellan at WalmartLabs, Marshfield Clinic, and Johnson Control. These are newer and still ongoing evaluations.

WalmartLabs deploy multiple EM systems for various purposes. As a first project, the EM team tried to debug a system that matches product descriptions. Since it is a complicated “blackbox” in production, they tried proxy debugging (Section 4.5). Specifically, they debugged a random forest based matcher and used the debugging result to clean the data, fix labels, and add new features. This significantly improved the system in production: increasing recall by 34% while reducing precision slightly by 0.65%. This indicates the promise of proxy debugging. In fact, 3 teams out of the 24 teams discussed in the previous subsection also used proxy debugging.

For Marshfield Clinic, we are currently helping to develop an EM workflow that uses learning and rules to match drug descriptions. Here labeling drug descriptions is very expensive, requiring domain experts who have limited time. They are also concerned about skewed data, i.e., too few matches in the current data. Taken together, this suggests that the sampling and labeling solution we discussed in Section 4.3 is well motivated, and we have been using a variant of that solution to help them label data. Yet another problem is that the Marshfield team is geographically distributed, so they would really like to have a cloud-based version of Magellan.

Finally, we are currently also working with Johnson Control to match data related to heating and cooling in buildings. The data that we have seen so far is very dirty. So the JCI team wants to extend Magellan with many more cleaning capabilities, in terms of Python packages that can immediately be made to work with Magellan’s data.

## **6.3 Summary**

Our experiments show that (a) current users can successfully follow the how-to guide to achieve high matching accuracy on diverse data sets, (b) the various tools developed for Magellan (e.g., debuggers) can be highly effective in helping the users, (c) practical EM requires a wide range of capabilities, e.g., cleaning, extraction, visualization, underscoring the importance of placing Magellan in an eco-system that provides such capabilities, and (d) there are many more EM challenges (e.g., cloud services) raised by observing Magellan “in the wild”.

## **7. DISCUSSION**

Our goal in this paper is not to show that we can develop a single EM management system (EMMS) that unifies all existing EM approaches. In fact, given the wide variety of existing EM approaches (that use a wide variety of EM workflows), we suspect it would be extremely difficult to build a single unifying EMMS.

Instead, our goal is to show that (a) it is important to go beyond EM algorithms to develop EM systems, (b) current EM systems have major limitations that prevent their widespread use in practice, (c) we can develop a methodology and architecture, as exemplified by Magellan, to build what we call “EM management systems” that address these

limitations, and (d) doing so also raises many novel research challenges.

Our hope is that the methodology and architecture of Magellan, as well as lessons learned building it, can be used as a “unifying template” to develop other EMMSs. We envision that each EMMS will address a set of related EM scenarios using a set of Python packages, but that the systems can seamlessly reuse a large portion of one another’s code and commands. (It is important to note that we do not think each EM scenario merits its own EMMS; an EMMS can address multiple EM scenarios, as we discuss at the end of this section.)

To make the above discussion more concrete, in what follows we will discuss how the methodology, architecture, and lessons of Magellan, which so far has focused on the EM scenario of matching two tables using learning and rules, can be applied to three additional EM scenarios: matching strings, linking a table into a knowledge base, and EM using iterative blocking.

**Matching Strings:** This is the problem of finding strings from a single given set or across two given sets that refer to the same real-world entity, e.g., “David Smith” and “Dave M. Smith”. This problem is a special case of EM, but due to its restrictive setting, it has typically been studied apart from EM, and numerous string matching solutions have been developed [28, 21].

Most string matching solutions focus on developing similarity measures (e.g., edit distance, Jaccard, TF/IDF, soft TF/IDF, etc) and scaling up matching a large number of string pairs. The latter is often studied under the topic “string similarity joins” or “set similarity joins” [30, 34]. To scale, many techniques called “filtering” have been developed, such as length filtering, prefix filtering, etc. For example, length filtering states that two strings _x_ and _y_ match only if their lengths satisfy a constraint. Given this property, we can build an index on the length of the strings, then use this index to quickly find string pairs that can possibly match.

Today string matching suffers from problems similar to those of EM, namely there are numerous matching algorithms but very few effective end-to-end string matching systems. In particular, many software packages exist that implement string similarity measures (e.g., SimMetrics [5], SecondString [4], Jellyfish [3], Abydos [1]), but surprisingly very few open-source packages exist that scale up these measures (Flamingo [2] is one such package). There is also no user guidance, e.g., to select a good string similarity measure and to debug the filtering and matching steps.

To address these problems, we advocate building endto-end string matching systems, and we believe that the methodology/architecture/lessons of Magellan can be applied here. Specifically,

1. First we consider a few common string matching scenarios. One such scenario is to match two large sets of strings _A_ and _B_ .

2. Next, we develop a how-to guide for this scenario. This guide proposes that the user matches _A_ and _B_ in two stages: development and production. In the development state the user tries to come up with an accurate string matching workflow. Similar to the current Magellan’s workflow (see Figure 9), this workflow consists

18

of cleaning/extracting/transforming, blocking, then matching (where blocking basically implements one or more filtering strategies).

3. To help the user develop this workflow, we can provide tools similar to those in Magellan. For example, we need a tool to sample sets _A_ and _B_ to produce two smaller sets _A_<sup>_′_</sup> and _B_<sup>_′_</sup> ; we need tools to help debug the blockers and matchers; and so on.

4. To help the user execute the workflow fast in the production stage, we will develop tools that scale up steps of the workflow, on a single machine or a cluster (using Hadoop or Spark).

Since the workflow for string matching described above is relatively similar to those of the current Magellan system, we can consider extending Magellan to this string matching scenario.

**Linking a Table into a Knowledge Base:** We now examine the problem of linking a table into a knowledge base (KB). A KB captures information about a particular domain. It typically consists of a taxonomy of concepts (that cover the domain), a set of instances for each concept, relationships among the concepts, and domain integrity constraints. Given a table and a KB, we want to find all pairs _x, y_ ) such that _x_ is a tuple in the table and _y_ is an instance in the KB and they refer to the same real-world entity.

For example, let _A_ ( _name, phone, address, affiliation_ ) be a table where each tuple describes a person. Let _K_ be a KB that contains a set of person instances (e.g., those of concepts such as _professor_ and _student_ ). Then we want to link each tuple in _A_ to the instance in _K_ (if any) that describes the same person.

A growing body of work (including some of our own [25]) has examined this EM scenario, as it arises in a growing number of applications (e.g., search, data integration, question answering, query interpretation).

We believe that the current Magellan solution can be applied to this problem, but it may also need to be extended. Specifically, we can proceed as follows:

1. Each concept in the KB _K_ is typically described using a set of attributes (e.g., “phone”, “organization”, etc for concept _professor_ ), so each instance is typically described using a set of attribute-value pairs. As such, we can extract all “person” instances from _K_ and store them in a relational table _B_ .

2. Our linking problem then reduces to matching tuple pairs between tables _A_ and _B_ , and a Magellan-like system can be applied to this problem.

3. If the above approach already produces sufficiently high EM accuracy (e.g., greater than a desired threshold), then we stop. Otherwise, we need to exploit KBspecific information to increase the accuracy. Many solutions to do this have been proposed, and we can consider implementing those solutions as extensions to the current Magellan.

For example, in a recent work [25] we have developed the following solution. Suppose the EM pipeline so far has predicted that a tuple _x_ matches an instance _y_ . To verify, classify _x_ into a node _C_ in the taxonomy

(e.g., “Academic Personnel”), then check if _y_ is an instance of a concept in the subtree rooted at _C_ . If not, then we can conclude that _x_ does not match _y_ . We can implement this solution (as well as others) as extensions to the Magellan’s pipeline considered so far.

Building on the above ideas, we propose to develop a tableto-KB EM management system. First, we will develop a how-to guide based on Steps 1-3 described above. This guide will subsume the how-to guide of the current Magellan, but significantly extend it. The new EM workflow will start with the current EM workflow of Magellan (which consists of cleaning/extracting/transforming, blocking, then matching), but extend it with steps that exploit KB-specific information to improve accuracy (as described above). We will still distinguish the development stage and the production stage. In the development stage the user can use all Magellan tools, but we will also develop tools specifically to help exploit KB-specific information.

While it is possible to extend the current Magellan to handle linking a table into a KB, we believe it is better to build this as a separate (though related) table-to-KB EM management system that addresses just this table-to-KB EM scenario. First, this system will already be quite complex. So separating it from the current Magellan makes it simpler to manage conceptually and implementation-wise.

Second, and more importantly, we suspect that a generic table-to-KB solution may not work well for all domains. For example, a solution that works well for social media may not work well for biomedicine, and vice versa. Thus, we may need to have a generic table-to-KB system and ways to help users customize this system to each domain of interest. This generic table-to-KB system can be implemented as a set of Python packages (which can rely quite heavily on the current Magellan packages).

**EM Using Iterative Blocking:** So far Magellan has considered EM scenarios that cleanly separate the blocking and matching steps. However, some EM scenarios, such as iterative blocking [33], interleave the two. The iterative blocking approach takes as input a table of tuples _A_ and outputs a partition of _A_ into groups such that all tuples within a group match and tuples across groups do not match. Briefly, this approach works as follows.

1. First, we use multiple blocking heuristics to partition _A_ into multiple blocks. For example, one heuristic partitions _A_ based on “zipcode”; another heuristic partitions _A_ based on “affiliation”. Note that a tuple from _A_ can end up in multiple blocks.

2. Next, for each block _D_ , we preprocess it, then apply a CER (i.e., “core entity resolution”) algorithm to partition _D_ into groups of matching tuples. Each such group forms a “super” tuple.

3. Next, we send the newly created “super” tuples to all the other blocks. The intuition is that if a block _B_ 1 has two tuples _s_ and _t_ , then by comparing them in isolation, we may not be able to decide that they match. However, if we have just applied the CER algorithm to a different block _B_ 2 and determined that _s_ matches _r_ , then we can send the super tuple ( _s, r_ ) to _B_ 1 and this time with the information from _r_ , we may be able to decide that ( _s, r_ ) matches _t_ (and thus _s_ matches _t_ ).

19

**Figure 10: The EM workflow for the scenario of matching using iterative blocking.**

4. Then we repeat Steps 2-3 again, until no new super tuples are created. At this point we can examine the groups in the blocks to produce the final partition of _A_ .

Figure 10 shows the high-level workflow of the above EM approach.

As described, in principle we can extend the current Magellan solution to incorporate this approach. First, the current Magellan assumes blocking will produce a set of candidate tuple pairs. We can extend blocking to produce a set of blocks (each of which is a set of tuples), to handle Step 1 (described above). Second, we can encapsulate Steps 2-3 in a matcher, which takes as input a set of blocks and outputs a final partition of table _A_ . As such, the workflow in Figure 10 reduces to the typical workflow of current Magellan shown in Figure 9.

In practice, we do not believe extending the current Magellan is a good idea. The iterative blocking approach is sufficiently different from the current EM approaches considered in the current Magellan system (which clearly separates out the blocking and matching steps) that it is best to place it in a new EM management system.

However, we should still be able to apply the same methodology/architecture/lessons in building Magellan to building this new EMMS. For example, we need to start with a concrete how-to guide that gives step-by-step instructions to the user, then consider how to reuse Magellan’s tools or build new tools to help the user do these steps.

For example, at the start, how do we know which blocking heuristics to use and how to debug these heuristics? Another important decision (in the development stage) is to select and debug the CER algorithm. The paper [33] describes an elegant iterative blocking framework. But this framework assumes a set of blocking heuristics and a CER algorithm have already been specified. The new EMMS should help the user make these decisions, which can have a great effect on the ultimate accuracy of the EM process. And in helping the user make these decisions, the new EMMS can reuse many tools provided by the current Magellan. For example, the Magellan tool to debug a blocker (described in Section 4.2) can also be used here to debug and find out which set of blocking heuristics to use.

Finally, we note that the iterative blocking algorithm works in a way that is similar to the way many EM-by-clustering algorithms work. Thus, when we build a clustering-based EMMS, we can also consider whether that EMMS can also naturally cover the iterative blocking algorithm.

**How Many EMMSs Do We Need?** The above discussion may give the impression that each EM scenario merits its own EMMS. We do not believe this should be the case. Instead, if a set of EM scenarios are naturally related, they all should be addressed in a single EMMS.

For example, the current Magellan can naturally handle EM scenarios that use supervised learning, rules, and a combination of both. (Note that each of these is actually a “group” of EM scenarios. For example, there are EM scenarios using supervised learning that aim for high precision, high recall, high F-1, etc.)

As another example, many clustering-based EM scenarios follow sufficiently similar algorithms that they should be grouped into a single EMMS. And this EMMS may be able to incorporate the iterative blocking scenario described earlier as well.

At the moment we do not yet know how many EMMSs we will ultimately need to cover most common EM scenarios. But we expect that over time, as we attempt to extend Magellan or build new EMMSs to cover new EM scenarios, this situation will become clearer. Further, as discussed earlier, we believe that the methodology, architecture, and lessons of Magellan can be applied to build these EMMSs. Finally, even though this paper has focused on EM, we believe that this methodology/architecture/lessons may also carry over to building systems that manage other kinds of problems, such as schema matching, IE, and data cleaning.

## **8. RELATED WORK**

Numerous EM algorithms have been proposed [16, 22]. But far fewer EM systems have been developed. We discussed these systems in Section 2.2 (see also [16]). For matching using supervised learning (Section 4), some of these systems provide only a set of matchers. None provides support for sampling, labeling, selecting and debugging blockers and matchers, as Magellan does.

Some recent works have discussed desirable properties for EM systems, e.g., being extensible and easy-to-deploy [19], being flexible and open source [15], and the ability to construct complex EM workflow consisting of distinct phases, each requiring a specific technique depending on the given application and data requirements [23]. These works do not discuss covering the entire EM pipeline, how-to guides, building on top of data analysis and Big Data stacks, and open-world systems, as we do in this paper.

Several works have addressed scaling up blocking (e.g., [18, 27, 32, 6]), learning blockers [12, 20], and using crowdsourcing for blocking [26] (see [17] for a survey). As far as we know, there has been no work on debugging blocking, as we do in Magellan.

On sampling and labeling, several works have studied active sampling [29, 9, 11]. These methods however are not directly applicable in our context, where we need a representative sample in order to estimate the matching accuracy (see Step 6 in Figure 3). For this purpose our work is closest to [26], which uses crowdsourcing to sample and label.

Debugging learning models has received relatively little attention, even though it is critical in practice, as this paper

20

has demonstrated. Prior works help users build, inspect and visualize specific ML models (e.g., decision trees [8], Naive Bayes [10], SVM [14], ensemble model [31]). But they do not allow users to examine errors and inspect raw data. In this aspect, the work closest to ours is [7], which addresses iterative building and debugging of supervised learning models. The system proposed in [7] can potentially be implemented as a Magellan’s tool for debugging learning-based matchers.

Finally, the notion of “open world” has been discussed in [24], but in the context of crowd workers’ manipulating data inside an RDBMS. Here we discuss a related but different notion of open-world systems that often interact with and manipulate each other’s data. In this vein, the work [13] is related in that it discusses the API design of the scikit-learn package and its design choices to seamlessly tie in with other packages in Python.

## **9. CONCLUSIONS & FUTURE WORK**

In this paper we have argued that significantly more attention should be paid to building EM systems. We then described Magellan, a new kind of EM systems, which is novel in several important aspects: how-to guides, tools to support the entire EM pipeline, tight integration with the PyData eco-system, open world vs. closed world systems, and easy access to an interactive script environment.

We plan to conduct more evaluation of Magellan, to further examine the research problems raised in this paper, to extend Magellan with more capabilities (e.g., crowdsourcing), and to deploy it on the cloud as a service. We will also explore managing more EM scenarios. In particular, we plan to extend Magellan to handle string matching, which uses workflows similar to those of matching using supervised learning. Other interesting EM scenarios include linking a table into a knowledge base (e.g., [25]) and matching using iterative blocking [33]. The former can potentially be incorporated into the current Magellan, but the latter will likely require a new EM management system (as it uses a very different kind of EM workflows).

**Acknowledgment:** We thank the reviewers for invaluable comments. This work is supported by gifts from WalmartLabs, Google, Johnson Control, and by NIH BD2K grant U54 AI117924.

## **10. REFERENCES**

- [1] Abydos. https://github.com/chrislit/abydos.

- [2] Flamingo. http://flamingo.ics.uci.edu/.

- [3] Jellyfish. https://github.com/jamesturk/jellyfish.

- [4] SecondString.

   - https://github.com/TeamCohen/secondstring.

- [5] SimMetrics.

   - https://github.com/Simmetrics/simmetrics.

- [6] F. N. Afrati, A. D. Sarma, D. Menestrina, A. Parameswaran, and J. D. Ullman. Fuzzy joins using MapReduce. ICDE, 2012.

- [7] S. Amershi, M. Chickering, S. M. Drucker, B. Lee, P. Simard, and J. Suh. Modeltracker: Redesigning performance analysis tools for machine learning. CHI, 2015.

- [8] M. Ankerst, C. Elsen, M. Ester, and H.-P. Kriegel. Visual classification: An interactive approach to decision tree construction. KDD, 1999.

- [9] A. Arasu, M. G¨otz, and R. Kaushik. On active learning of record matching packages. SIGMOD, 2010.

- [10] B. Becker, R. Kohavi, and D. Sommerfield. Visualizing the simple Bayesian classifier. In _Information Visualization in Data Mining and Knowledge Discovery_ , 2002.

- [11] K. Bellare, S. Iyengar, A. G. Parameswaran, and V. Rastogi. Active sampling for entity matching. KDD, 2012.

- [12] M. Bilenko, B. Kamath, and R. J. Mooney. Adaptive blocking: Learning to scale up record linkage. ICDM, 2006.

- [13] L. Buitinck et al. API design for machine learning software: experiences from the scikit-learn project. _arXiv preprint arXiv:1309.0238_ , 2013.

- [14] D. Caragea, D. Cook, and V. Honavar. Gaining insights into support vector machine pattern classifiers using projection-based tour methods. KDD, 2001.

- [15] P. Christen. Febrl: A freely available record linkage system with a graphical user interface. HDKM, 2008.

- [16] P. Christen. _Data Matching_ . Springer, 2012.

- [17] P. Christen. A survey of indexing techniques for scalable record linkage and deduplication. _IEEE TKDE_ , 24(9):1537–1555, 2012.

- [18] X. Chu, I. F. Ilyas, and P. Koutris. Distributed data deduplication. _PVLDB_ , 9(11):864–875, 2016.

- [19] M. Dallachiesa, A. Ebaid, A. Eldawy, A. Elmagarmid, I. F. Ilyas, M. Ouzzani, and N. Tang. Nadeef: A commodity data cleaning system. SIGMOD, 2013.

- [20] A. Das Sarma, A. Jain, A. Machanavajjhala, and P. Bohannon. An automatic blocking mechanism for large-scale de-duplication tasks. CIKM, 2012.

- [21] A. Doan, A. Halevy, and Z. Ives. _Principles of Data Integration_ . Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 1st edition, 2012.

- [22] A. K. Elmagarmid, P. G. Ipeirotis, and V. S. Verykios. Duplicate record detection: A survey. _IEEE TKDE_ , 19(1):1–16, 2007.

- [23] M. Fortini, M. Scannapieco, L. Tosco, and T. Tuoto. Towards an open source toolkit for building record linkage workflows. In _In Proc. of the SIGMOD Workshop on Information Quality in Information Systems_ , 2006.

- [24] M. J. Franklin, D. Kossmann, T. Kraska, S. Ramesh, and R. Xin. CrowdDB: answering queries with crowdsourcing. SIGMOD, 2011.

- [25] A. Gattani et al. Entity extraction, linking, classification, and tagging for social media: A Wikipedia-based approach. _PVLDB_ , 6(11):1126–1137, 2013.

- [26] C. Gokhale, S. Das, A. Doan, J. F. Naughton, N. Rampalli, J. Shavlik, and X. Zhu. Corleone: Hands-off crowdsourcing for entity matching. SIGMOD, 2014.

- [27] L. Kolb, A. Thor, and E. Rahm. Dedoop: efficient deduplication with Hadoop. _PVLDB_ , 5(12):1878–1881, 2012.

- [28] G. Navarro. A guided tour to approximate string matching. _ACM Comput. Surv._ , 33(1):31–88, Mar. 2001.

21

- [29] S. Sarawagi and A. Bhamidipaty. Interactive deduplication using active learning. KDD, 2002.

- [30] S. Sarawagi and A. Kirpal. Efficient set joins on similarity predicates. In _Proceedings of the 2004 ACM SIGMOD International Conference on Management of Data_ , SIGMOD ’04, pages 743–754, New York, NY, USA, 2004. ACM.

- [31] J. Talbot, B. Lee, A. Kapoor, and D. Tan. Ensemblematrix: Interactive visualization to support machine learning with multiple classifiers. CHI, 2009.

- [32] R. Vernica, M. J. Carey, and C. Li. Efficient parallel set-similarity joins using MapReduce. SIGMOD, 2010.

- [33] S. E. Whang et al. Entity resolution with iterative blocking. SIGMOD, 2009.

- [34] M. Yu, G. Li, D. Deng, and J. Feng. String similarity search and join: a survey. _Frontiers of Computer Science_ , pages 1–19, 2015.

22

