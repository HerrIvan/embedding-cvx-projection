# Embedding-Cvx-Projection

This repo introduces a simple [convex optimization](https://en.wikipedia.org/wiki/Convex_optimization) method that aims to find the best approximation to a target embedding based on a reduced set of relevant but diverse embeddings.

The operation achieved is analogous to answering the question: _how can I best explain my input embedding by *meaningfully* combining embeddings from a specified set?_.

Presumed applications for it are:
- retrieving relevant but diverse contexts in the retrieval step of [Retrieval Augmented Generation](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) (RAG) workflow.
- extending [knowledge graphs](https://en.wikipedia.org/wiki/Knowledge_graph) by inferring relationships from already existing ones.

For an _in-depth_ discussion regarding its derivation and applications, please check the extended [background](docs/background.md) document that discusses the theory, implementation, and examples.


## HOW TO USE IT

### Installation

Before running any examples, you'll need to install the package from root directory of this repository. There you can run:

```
pip install .
```
This will install the `embedding_cvx_projection` package along with its dependencies.

### Running examples

Once the package is installed, you can run the example scripts to understand how to  use this technique. Indeed, the `embedding_cvx_projection` package is very lean, so that this repository can be best understood as documentation and examples in how to use the given `cvxpy` projection technique.

#### Common nouns example

##### Overview

This script explores a scenario mimicking the extension of a Knowledge Graph (KG) by linking additional features to an entity based on already linked features. 

For instance, an e-commerce shop selling furniture may have features associated to products, where the features are drawn from a closed but extensive set. In a specific example, say a furniture piece has properties _waterproof_ and _durable_; leveraging the inference mechanism implemented in our algorithm, it would possible to automatically attach (or suggest attaching) _outdoors_ to it.

⚠️ **Note**: The given algorithm uses OpenAI embeddings, so make sure you have a valid `OPENAI_API_KEY` set up for on-the-fly embedding if any of your words are not in the list of 500 common nouns.

##### Task

The goal here is to identify which nouns, from a predefined list of around 500 common nouns, are closely related or can be derived based on a given list of input words. This algorithm utilizes the OpenAI embeddings for those 500 nouns, encapsulated in an `EmbeddingLibrary` object.


##### Running it 

Before running this example you need to install the modules in `requirements.txt`. 

Navigate to `examples/common_nouns` directory, and execute the script `infer_nouns_from_nouns.py` with your input words as arguments:
```
python infer_nouns_from_nouns.py word1 word2 ...
```

The script processes then processes input words which can either belong to a list of 500 common nouns or not. If not, they can be embedded on-the-fly using the OpenAI embeddings endpoint. For on-the-fly embeddings, a valid `OPENAI_API_KEY` environment variable needs to be set.

##### Output 

Here's an example output for the inputs `mountain`, `summit`, `explosion`, and `fire`:

    word, 	  value, 	mixture, 		gain
    volcano,  0.918, 	[0.38 0.1  0.35 0.17], 	0.377
    lava, 	  0.893, 	[0.33 0.08 0.31 0.28], 	0.330
    impact,   0.887, 	[0.17 0.23 0.29 0.32], 	0.317
    bomb, 	  0.909, 	[0.11 0.13 0.38 0.37], 	0.310

- The first column lists words whose embeddings are better approximated given the embeddings of the input words. 
- The second column measures the goodness of the approximation (cosine similarity minus the regularization term). 
- The third column provides the mixing recipe, indicating the proportions in which input embeddings should be combined to approximate the target embedding more closely. 
- The fourth column shows the gain in similarity, indicating how much closer the target embedding is to the combined embedding than to any _simple_ embedding in the set.

📈 **Interpreting the Output**: In the example output, the word 'volcano' is well approximated because it has a high value of 0.918 and has a mixture of all the input words where 'mountain' and 'explosion' have higher weights. The 'gain' indicates that combining the input words brings us 0.377 closer to 'volcano' than any single input word would.

'Bomb' is also well approximated, but in this case, the words weighted higher are 'explosion' and 'fire'.

#### Synthetic retrieval example

##### Overview

This example simulates a scenario of matching a query embedding against a set of embeddings. That is, the same operation that may be performed in the *R*etrieval step of a Retrieval Augment Generation (RAG) workflow. In that case the embeddings retrieved will point to _contexts_ (text document or chunks thereof) that may contain relevant information with respect to the input query. In practice, what we expect from applying our method to that end is retrieving a set of relevant and diverse embeddings, with which to construct a compact but relevant and informative context for the downstream Generation step.

🔍 **What to Expect**: This example aims to mimic a real-world retrieval scenario but in a simplified manner. It helps in evaluating the effectiveness of our algorithm in returning diverse and relevant embeddings.

##### Task

Instead of using embeddings extracted from real texts, here we abstract the problem generating random vector embeddings with a particular structure. Namely, embeddings are generated around four prototype embeddings, therefore generating four clusters. Moreover, those clusters are exponentially unbalanced in size, with size (number of members) 1, 9, 90 and 900. 

As a final step, the query embedding is constructed also as a random additive combination of the four prototype embeddings. With that, we expect that in order to better reconstruct input embedding, one should gather at least one embedding from each of the four clusters. 

The reason for that, is that one you have retrieved an embedding from one cluster, even though other embeddings from that cluster may still lie close to the query embedding, returning them will result mostly in redundant information that would because of its redundancy will hinder the performance of the generation step.

📚 **Why This Matters**: The example demonstrates how redundant or imbalanced clusters can affect the quality of retrieved information. The goal is to showcase the capability of the algorithm to provide a diverse set of relevant embeddings.

##### Running it


Navigate to `examples/synthetic_retrieval` and run:

```
python redundant_unbalanced_contexts.py
```

You can use different parameter to configure the problem, controlling how disperse are the clusters and how balanced are they combined in the input query. 

```
python redundant_unbalanced_contexts.py
   -p: number of problems to run
   -n: number of items to retrieve
   -c: minimum contribution of each cluster
   -d: cluster dispersion
```

🛠 Customization: Feel free to experiment with the parameters to see how the algorithm performs under different conditions. For example, a higher cluster dispersion might make the problem more challenging.


Note that if the cluster dispersion is set very high (>0.5) cluster structure may eventually be lost.

##### Output

The script outputs the average number of contexts present in the solution, along with a summary of the configuration parameters. 

```
Problem conf. summary:
----------------------
Number of problems run: 40
Number of items retrieved for each the solution: 5
Minimum contribution of each cluster to the solution: 0.3
Cluster dispersion: 0.1
----------------------
Different clusters per solver (average and std):
solve_by_cvx_projection: mean 2.75, std: 0.54
solve_by_least_squares: mean 2.225, std: 0.88
solve_by_similarities: mean 1.325, std: 0.52
```

📊 **Analysis**: The average number of different clusters returned by solve_by_cvx_projection is here consistently higher, indicating that our method is effectively capturing diversity, which would be beneficial for downstream tasks like information retrieval or question answering.

## Summary

This repository presents a convex-optimization method designed for approximating target embeddings using a reduced set of diverse and relevant embeddings. Utilizing cosine similarity as the measure of closeness, the method projects a given embedding onto a convex cone defined by a collection of other embeddings. 

### _Is this method novel?_

While the method and its application in the domain of reasoning with embeddings may not be groundbreaking, it addresses an important gap in practical implementation. Convex optimization techniques are often buried in technical jargon, making them less accessible to practitioners. This very simplicity and obviousness are precisely what make these techniques powerful yet underutilized, particularly as we see a surge in embedding adoption through Language Models and Vector Databases.

### _Why this repo matters_

This repository aims to demystify these robust, mathematically sound techniques by offering approachable practical examples. By doing so, it seeks to accelerate community adoption and enhance current pipelines, providing a valuable resource even if the methods themselves are not novel. The focus here is not on reinventing the wheel but on facilitating its more effective use in a broader context.

**Note**: indeed, at this stage, depending on your use case, rather than importing this module, it will be leaner for your project to directly import `cvxpy` and simply add the code with the problem definition ([here](src/embedding_cvx_projection/cone_projection.py)).

### _Next steps_

There mostly two ways forward for this repo: 

- *Additional use cases*: Develop a more comprehensive set of use case examples, for instance, with a real RAG workflow or specific relation inference from known ontologies. 

- *Integrations*: Explore integrations either with existing vector DBs or an application with a specific Knowledge DB (e.g. an integration with a neo4j graph DB.)

Contributions are welcome.










