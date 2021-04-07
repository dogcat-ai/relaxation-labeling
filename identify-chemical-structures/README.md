# The problem
For this Kaggle competition, the participants are asked to use a computer vision algorithm to derive a chemical formula from an image depicting a chemical structure.  The formula must be given in the InChl format.  The image depicting the chemical structure will most likely be drawn with standard techniques, but the author of this README does not have the knowledge to fully enumerate these techniques at this present time, though he knows a few basic components of these images.

# The solution
We plan on using a combination of Relaxation Labeling and Optical Character Recognition to identify all salient features of an image.  The identification method will not only generate a list of the elements and bonds found in the image, but will also identify the connections between all of these objects; that is, it will precisely describe the chemical structure depicted in the image, in the form of a graph (in the graph theory sense of what a graph is).  From this graph, the chemical formula in InChl format should be easy to derive.

Relaxation Labeling will be looking mostly for bonds, in the form of line segments of fairly-predictable length and orientation, while the Optical Character Recognizer will be looking for small sets of alphanumeric characters, representing elements or simple molecules.  The Relaxation Labeling Process will use laws of chemistry to constrain the set of elements that can exist at some bond endpoint, and which sets of elements can exist together as simple molecules.  It will simultaneously constrain the types of bonds that can exist between two elements (or between two simple molecules).

Let us begin with an example of a small-ish molecule, to illustrate our general approach.  (Note: the molecule depicted is Serotonin--a chemical which incidentally increases in direct proportion to the time the author spends with his typist.)


                           NH2
                         _/
             HO         /
               \ / \\ /\\
                ||   |  /
                 \ // -N
                       H

So, ascii is a pretty shitty medium for depicting chemical structures, but... you get what you pay for.  If this were an actual image of Serotonin used in pharmacology textbooks, you would see a cleanly drawn hexagon, to the right of it a cleanly drawn pentagon, plus the offshoots.  Notice that there are four double bonds in this picture.  There are three simple molecules explicitly shown: HO, NH2, and NH; at each remaining vertex, carbon is implicit.  Notice that there are four double bonds in this picture.  There are three chemicals explicitly shown: HO, NH2, and NH.  It is implicit that Carbon is located at each vertex where there is not a chemical explicitly denoted.  Typically, a Hydrogen bond is also implicit, when it will give a particular Carbon 4 bonds.

So, what are the different labels we can enumerate for our problem?

First, there are two broad categories: (1) alphanumeric characters, and (2) bond symbols.

    (1) Alphanumeric characters | Though there are 118 elements, there are only 26 letters in the alphabet, and 10 digits.  So each alphanumeric character has 26+26+10=62 possible labels (26*2 because uppercase and lowercase), at the finest-grain resolution.  (In fact, we will probably have fewer than 62 labels at this resolution, because the union of all elements in the periodic table will probably exclude some characters of the alphabet--capital, lowercase, or both, depending on the letter.) At the next level of resolution (semantically broader), we have individual elements, from the periodic table.  We have 118 possible labels, at this resolution.
    (2) Bond symbols | There are a few different symbols indicating different kinds of bonds, and different combinations of bonds.  There are single lines, indicating a single bond; double lines, indicating a double bond; hatched lines that form a triangle, indicating a bond going away from the reader (in the Z direction); and a solid black triangle, indicating a bond going toward the reader (in the Z direction).  So each bond symbol has 4 possible labels, at the finest-grain resolution.  The 2D bonds (the single lines or the double lines) can form a ring such as a hexagon or a pentagon.  At this next level of resolution (semantically broader), we have 3 labels: hexagonal rings, pentagonal rings, and "lone segments" (lone segments include single bonds, double bonds, and protruding and receding bonds, but these bonds are not part of a ring; an alternative title could be "degenerate rings" or "non-rings").  There is also a sort of "null bond", which exists when the bond is not drawn out (for example, HO in the depiction of Serotonin above), but I am not sure how to deal with this yet, so I will leave it aside.

If we are to implement "Hierarchical Relax", the hierarchy will look like this:

I . Alphanumeric representation of elements or simple molecules
      A. Simple molecules
           1. Elements
                a. Letters
           2. Numbers
                a. Digits
      B. Elements  
II. Bond symbols
      A. Hexagon
           1. Single bond
           2. Double bond
      B. Pentagon
           1. Single bond
           2. Double bond
      C. Lone bond
           1. Single bond
           2. Double bond
           3. Protruding Hydrogen bond
           4. Receding Hydrogen bond

