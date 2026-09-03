import { ChatOpenAI } from 'langchain/chat_models/openai'
import { PromptTemplate } from 'langchain/prompts'
import { StringOutputParser } from 'langchain/schema/output_parser'
import { retriever } from './utils/retriever.js'
import { combineDocuments } from './utils/document.js'
import { RunnablePassthrough, RunnableSequence } from "langchain/schema/runnable"

const openAIApiKey = process.env.OPENAI_API_KEY
const llm = new ChatOpenAI({ modelName: 'gpt-3.5-turbo-1106', openAIApiKey, maxTokens: 3000})

// standalone prompt template
const standaloneQuestionTemplate = 'Given a question, convert it to a standalone question. question: {question} standalone question:'
const standaloneQuestionPrompt = PromptTemplate.fromTemplate(standaloneQuestionTemplate)

// Q&A prompt template
const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Nimbus Cloud based on the context provided. Try to find the answer in the context. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@nimbuscloud.dev. Don't try to make up an answer. Always speak as if you were chatting to a friend.
context: {context}
question: {question}
answer: `
const answerPrompt = PromptTemplate.fromTemplate(answerTemplate)

// chains
const standaloneQuestionChain = RunnableSequence.from([
    standaloneQuestionPrompt,
    llm,
    new StringOutputParser()
]);
const retrieverChain = RunnableSequence.from([
    prevResult => prevResult.standalone_question,
    retriever,
    combineDocuments
]);
const answerChain = RunnableSequence.from([
    answerPrompt,
    llm,
    new StringOutputParser()
]);

// main chain: standalone question -> retrieve + combine docs as context -> answer
const chain = RunnableSequence.from([
    {
        standalone_question: standaloneQuestionChain,
        original_input: new RunnablePassthrough()
    },
    {
        context: retrieverChain,
        question: ({ original_input }) => original_input.question
    },
    answerChain
]);

const response = await chain.invoke({
    question: 'I have 3 questions: (a) what makes Nimbus Cloud good?; (b) how fast is Nimbus Cloud?; (c) which server locations are available?; (d) can I copy-paste into the console, and how?'
})

console.log(response)