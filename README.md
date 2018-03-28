# story-generator
IN PROGRESS

Train a recurrent language model on a corpus of short stories and then generate new short stories.

Optionally, uses a decoupled neural interface to shorten the effective sequence length and speed up training by estimating the hidden state at intermediate points in the story.

TODO:
1. Add the option of providing a pre-trained embedding
2. Finish coding the post-training sampling function to generate story
