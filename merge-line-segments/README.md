In order to merge line segments within the Relaxation Labeling framework, we simply need to say whether one particular line segment belongs with another particular line segment.  In the molecule we are currently dealing with, we want to have 7 line segments in the end.  We currently have 15, I believe.  One way to make sure we have enough labels is to start off with 15 labels, and let some object-labels be the "anchor" for other object-labels.  In fact, we could do something like this:

I.  Object 1
	A. Label 1
	B. Label 2
	C. Label 3
	...
	N. Label 14
	O. Label 15
II. Object 2
	A. Label 1
	B. Label 2
	C. Label 3
	...
	N. Label 14
	O. Label 15
...
XV. Object 15
	A. Label 1
	B. Label 2
	C. Label 3
	...
	N. Label 14
	O. Label 15

This may be a fine way to implement this application of the Relaxation Labeling Process.  However, if we can have fewer labels, that makes the algorithm more computationally inexpensive.  An alternative implementation is:

compatibility(object i, label SAME, object k, label SAME)
compatibility(object i, label DIFFERENT, object k, label DIFFERENT)

I'm not quite sure how I could implement the above, though...
