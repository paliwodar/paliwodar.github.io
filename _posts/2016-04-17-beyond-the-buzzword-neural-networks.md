---
layout:     post
title:      Beyond the Buzzword&#58; Neural Networks
date:       2016-04-16
summary:    So, What are Neural Networks?
categories: neural-networks machine-learning coursera programming
---

In this article I'm gonna share some thoughts[^disclaimer] of mine about neural networks[^nn]. In my opinion one of the objects that are simultaneously both 
the most over- and underestimated in computer science. It's a kind of things everyone (these days more and more often) talks about but only 
a few know the details. 

That is also the case of mine. Or actually was. I've never had a good enough reason to gain more knowledge.
Every time I ended with a brief image of a dosen of circles visualizing some kind of a flow. I also used to see 
people reacting with true admiration each time somebody mentioned he built some leaf shape based plant species recognizer. 
Of course using you know what. Wow, he knows neural networks, he must be extremely clever, isn't he? Well, it was hard to say for me. 
I was somewhere in between, lacking the knowledge required to verify anything. 

## Coursera

But now things' changed thanks to [Machine Learning](https://www.coursera.org/learn/machine-learning) course which I solemnly advise to try. 
Incidentally I would like to point that it is provided by Andrew Ng -- a Coursera co-founder
and well-known person in the industry. It's just like learning Scala in Martin Odersky's class.

As a motivation I recommend you to take a look at [AlphaGo](http://deepmind.com/alpha-go.html) page or to see 
[how it's won with European Go champion](https://www.youtube.com/watch?v=g-dKXOlsf98)[^european_champion]. 
So they built a remarkable Go engine. And using not just one, but two[^two_networks] deep neural networks! Ok, I bet now everybody is eager to master this stuff. Great, let's go for it.

## Genesis

So neural networks idea is nature-inspired. This seems to be very common approach for dealing with very complex problems in computer science.
Everybody must have heard about genetic algorithms or ant colony optimization for instance. 
In our case, as the name suggests the model tries to mimic the brain neurons behaviour. 
But for me, the brain stuff could be easily only an interpretation of NN model. 
It's because it looks like a smooth extension of logistic regression. I know this can sound a bit vague. Let's delve into details.

## Regression analysis

This term is basically about estimating relationships among variables. Variables can mean anything. In most cases they are
representing features of an object we are analysing. 

In case of Go we could have \\(x_{i,j} \in \\{0,1,2\\}\\)  for \\(i, j = 1,...,19\\) 
representing the state of the board[^go_board] and \\(y_1, y_2\\) describing coordinates of the next move.
Obviously there is a correlation between what's on the board and what should be played next, isn't there?

## Linear regression

Let's have however a little bit simpler example, perhaps the most popular one when it comes to talk about NN.
House Pricing! Assume we have two features: \\(x_1\\) and \\(x_2\\) representing area in square meters and distance from
the city centre in kilometers. And of course the price itself. Our task is to estimate the latter.

Perhaps no complicated formulas are necessary. 
Let's try the simplest thing possible, like adding the values of the features. As simple as that, why not? 
Unfortunately it doesn't sound like adding 50 square meters and 2 kilometers can give 1000$. So why don't we introduce some factors?
And maybe some bias? With this approach we get the following hypothesis function[^regression_equation]:

$$ h_\Theta(x) = \Theta_0 + x_i\Theta_1 + x_2\Theta_2 $$

Not so sure if this is the way they invented it. It can make some sense on the other hand. 
By the way, please note that the formula above can be written as \\(\sum_{i=0}^2 x_i\Theta_i\\), where \\(x_0 = 1\\) which 
is equivalent to a multiplication of vectors: \\(x^T\Theta\\). Actually I must admit this course is all about
matrix multiplication. At least the programming part which involves Octave. Not that I am complaining. It's fun!

So the only thing we need now is to find proper values of \\(\Theta\\). 
It's the moment some training data and some fairly understable calculus come into play.
Fortunately we can skip the details. The basic idea is to minimize the error between predicted prices and the training data we have collected.

After obtaining \\(\\Theta\\) we get a hyperplane[^hyperplane] equation which gives us the estimation we are looking for. 
Here you can see a line that predicts a house price basing on its area:

![Linear regression](/images/linear_regression.svg){: .center-image .scaled-down-70-image}

So basically, what we've just more or less accurately described above is called a linear regression.

## Logistic regression

Logistic regression is a simple adaptation that allows us to deal with problems of discrete solution spaces, 
like deciding if it's gonna rain today or not basing on some simple weather information[^one_versus_all].

So now our result should be \\(0\\) or \\(1\\). But our hypothesis function can give us any number.
Fortunately it could be solved with pretty much the same math as before. But now we would assume the result is \\(1\\) if only \\(h_\Theta(x) \ge 0\\). 

It's convenient to make a use of so called *sigmoid* function:

$$ g(t) = \frac{1}{1 + e^{-t}}  $$

That looks like this:

![Logistic curve](/images/logistic-curve.svg){: .center-image .scaled-down-50-image}

It's easy to see that its result is close to one and zero for high and low input values correspondingly.

So the logistic regression formula would be \\(h_\Theta(x) = g(x^T\Theta)\\). We can now interpret the result as a probability that the event occurres,
assuming the result is \\(1\\) if we get a value higher than \\(0.5\\).

This kind of functional form is commonly called a single-layer perceptron.

## Simple Boolean functions

Please be patient. We are almost there. Neural Networks are just around the corner. You can believe me you'll be surprised!

We'll take a look now at example functions that are expressed using logistic regression. Let's say a boolean negation, AND and OR.
The simplest functions ever one would say. However not necessarily from this perspective.

#### Negation

We have here one input variable \\(x\in\\{0,1\\}\\). The logistic regression formula would be quite simple:
$$ g(\Theta_0 + x\Theta_1) = \neg x $$.

So we should basically have:

$$g(\Theta_0 + \Theta_1) = 0 \quad g(\Theta_0) = 1$$

which by definition of *sigmoid* function is equivalent to:

$$\Theta_0 + \Theta_1 < 0 \quad \Theta_0 > 0$$

It's very easy to find proper \\(\\Theta\\) values now. It can be for instance: \\(\Theta = [10, -20]\\).
We could visualize the above with the following graph:

![Negation perceptron](/images/negation.svg){: .center-image .scaled-down-45-image}

#### AND 

After the warm up we can go directly to the point here. 

<!-- For the sake of completeness let's take a look at the table describing logical conjunction behaviour:

$$ \begin{array}{|c|c|c|}
		\hline
		x_1 & x_2 & x_1 \land x_2  \\
		\hline	
		1 & 1 & 1 \\
		1 & 0 & 0 \\
		0 & 1 & 0 \\
		0 & 0 & 0 \\
		\hline	
\end{array} $$

-->
So we need to have \\(\Theta_0 + x_1\Theta_1 + x_2\Theta_2 = x_1\land x_2\\) for all the possible cases.
The example solution here can be: \\(\Theta = [-30, 20, 20]\\). We can see that \\(20x_1 + 20x_2 - 30 > 0\\) iff[^iff] \\(x_1 = 1\\) and \\(x_2=1\\).

The graph that expresses that:

![And perceptron](/images/and.svg){: .center-image .scaled-down-45-image}

#### OR

We can skip the details here. Let's draw however a visualisation for logical alternative:

![Or perceptron](/images/or.svg){: .center-image .scaled-down-45-image}

## A bit more demanding function: XOR

XOR a.k.a. exclusive disjunction is a boolean operation that returns true iff the operands differ. Its truth table:

$$ \begin{array}{|c|c|c|}
		\hline
		x_1 & x_2 & x_1 \oplus x_2  \\
		\hline	
		1 & 1 & 0 \\
		1 & 0 & 1 \\
		0 & 1 & 1 \\
		0 & 0 & 0 \\
		\hline	
\end{array} $$


Ok, we've just gone through pretty basic stuff. It was necessary however to illustrate the big picture.

In addition to what has been said XOR is a canonical example of a function that cannot be solved with the simplest linear decision boundary. 
It is easy to observe that while looking at the chart:

![Xor_chart_empty](/images/xor_chart_empty.svg){: .center-image .scaled-down-50-image}

#### Polynomial regression

There are no line that could separate the data correctly. What can we do then in order to model XOR?
We could of course introduce more features using a polynomial regression. 

The example equation of a non-linear boundary can be for instance:
 $$ 2x_1 + 2x_2 - 5x_1x_2 - 1 > 0 $$ --- presented on the chart below:

![Xor_chart](/images/xor_chart.svg){: .center-image .scaled-down-50-image}

#### Putting things together

We can try however to figure out some alternative way of solving the XOR problem. It is required to take a closer look at the definition
of this boolean operation. And it looks as follows:

$$ 
\begin{align*}
x_1 \oplus x_2 &\equiv \neg (x_1 \land x_2) \land (x_1 \lor x_2)   %\\
	%&\equiv (\neg x_1 \lor \neg x_2) \land (x_1 \lor x_2)
\end{align*} $$

Let's try now combine what we already have in order to express the behaviour of XOR.
We should aim obtaining something like this:

![Xor gates](/images/gates.svg){: .center-image .scaled-down-40-image}

That's just a schema that uses logical gates. Let's then try to convert it to perceptrons language.
It seems that the last required thing is to note that \\(\Theta_{\text{NAND}} = -\Theta_{\text{AND}}\\). 

We have then all the pieces we need to build... a Neural Network!

![Xor](/images/xor.svg){: .center-image .scaled-down-70-image}

## Representation

Finally we reach that point! That is the essence of what we were pursuing in this journey. 
So it looks like a pretty straightforward thing, doesn't it?
I like the idea that NN are a lot like simple logistic regression based on features
that has been discovered by the network itself!

So it really behaves like a human neural system. We could perceive the input parameters
as readings from our body sensors, like fingertips or taste receptors.
It is then processed by the net of neurons that is being tweaked as different impulses are received.
Which also happens to an artificial network during the learning process.

We won't go into the further details here. Hope that is easy now to continue with some other sources 
once we understand the basics. Let's take a look however at the representation of such a network.
It turns out it is natural to represent \\(\Theta\\) as an array of matrices -- each corresponding to
different layer of the network. In the network above it would be:

$$

\begin{align*}
\Theta^{(1)} &=    \begin{bmatrix}
    \color{purple}{30} & \color{olive}{-10} \\
    \color{purple}{-20} & \color{olive}{20} \\
    \color{purple}{-20} & \color{olive}{20} 
    \end{bmatrix} \\
\Theta^{(2)} &=  \begin{bmatrix}
    \color{teal}{
    -30 \\
    20 \\
    20 }
    \end{bmatrix}
\end{align*}
$$


We could express the whole computation with the following equation[^bias_skipped]:

$$ h_\Theta(x) = g\left(g\left( x^T \Theta^{(1)} \right) \Theta^{(2)}\right)
$$

## Conclusion

It is necessary to mention that using a neural network that is already trained is fairly easy.
Since we have the values on the edges it is enough to perform a so called forward propagation.
The non-trivial thing however is the process of obtaining proper values.
There is some more or less advanced calculus required to fully understand this stuff
and because of that nobody can say it is a ridiculously easy topic.
Describing it goes (un)fortunately beyond the scope of the article and lies on the far side of author's math competences.

Thank you for your attention. Hope you enjoyed the reading!


---
[^disclaimer]: __Disclaimer__: The article can be strongly opinionated. Some minor factual erros can appear mostly due to an oversimplification. Moreover one will very unlikely see here strict definitions of things. He would rather face the author's feelings about what the things could be. 
[^nn]: Can be also referred as NN in this article.
[^two_networks]: Technically there might be much more.
[^go_board]: The board in Go has \\(19\times 19\\) crossing points the stones can be put on. The values \\(0, 1, 2\\) would mean no stone, black or white correspondingly.  
[^regression_equation]: Of course the formula doesn't have to look like this. It can be of many different forms like for instance some polynomial regression one: \\(h_\Theta(x) = \Theta_1 x_1^2 + \Theta_2 x_2^2\\). __Request__: If you spot any math that does not make much sense, feel encouraged to give the author a shout immediately!
[^one_versus_all]: In order to perform a classification with more than two classes we can use for example so called [one versus all](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest) strategy.
[^hyperplane]: Hyperplane is a subspace of one dimension less than its ambient space. So basically for 2-dimensional space (area and price) it is a line, while if the space is 3-dimensional (area, distance and price) its hyperplanes are 2-dimensional planes.
[^iff]: *iff* \\(\equiv\\) *if and only if*
[^bias_skipped]: Bias values have been skipped here to make the equation simpler. 
[^european_champion]: The information is still relevant however lately even more astonishing thing happened: AlphaGo has won 4:1 with one of the best modern Go players -- Lee Sedol.
