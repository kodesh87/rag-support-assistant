import { ChatOpenAI } from 'langchain/chat_models/openai'
import { PromptTemplate } from 'langchain/prompts'
import { StringOutputParser } from 'langchain/schema/output_parser'
import { retriever } from '../utils/retriever.js';
import { combineDocuments } from '../utils/document.js';
import dotenv from 'dotenv';

dotenv.config();

const openAIApiKey = process.env.OPENAI_API_KEY
const llm = new ChatOpenAI({ openAIApiKey })

const standaloneQuestionTemplate = 'Given a question, convert it to a standalone question. question: {question} standalone question:'
const standaloneQuestionPrompt = PromptTemplate.fromTemplate(standaloneQuestionTemplate)

const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given
question about Nimbus Cloud based on the context provided. Try to find the answer in the context. If you
really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the
questioner to email help@nimbuscloud.dev. Don't try to make up an answer. Always speak as if you were
chatting to a friend.
context: {context}
question: {question}
answer:
`;
const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);

// NOTE: this pipes the retrieved context straight into a PromptTemplate without
// ever calling the LLM on it — kept as-is, it's an early iteration that predates
// the working chain in index.js/server.js.
const chain = standaloneQuestionPrompt.pipe(llm).pipe(new StringOutputParser()).pipe(retriever).pipe(combineDocuments).pipe(answerPrompt);

const response = await chain.invoke({
    question: 'What are the technical requirements for running Nimbus Cloud? I only have a very old laptop which is not that powerful.'
})

console.log(response)