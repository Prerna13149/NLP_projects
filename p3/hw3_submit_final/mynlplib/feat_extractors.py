class SimpleFeatureExtractor:

    def get_features(self, parser_state):
        """
        Take in all the parser state information and return features.
        Your features should be autograd.Variable objects of embeddings.

        :param parser_state the ParserState object for the current parse (giving access
            to the stack and input buffer)
        :return A list of autograd.Variable objects, which are the embeddings of your
            features
        """
        # STUDENT
        # hint: parser_state.stack_peek_n
        stack_top = parser_state.stack_peek_n(1)[0]
        _, _, embed_stack_top = stack_top
        #print("embed_top: ", embed_stack_top)
        _, _, embed_buffer_top = parser_state.input_buffer_peek_n(1)[0]
        # _, _, embed_buffer_top = stack_top
        _, _, embed_buffer_second = parser_state.input_buffer_peek_n(2)[1]
        #emded_model = neural_net.VanillaWordEmbedding(test_word_to_ix, TEST_EMBEDDING_DIM)
        #embed_top = 
        return [embed_stack_top, embed_buffer_top, embed_buffer_second]

        # END STUDENT
