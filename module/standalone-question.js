import { ChatOpenAI } from "langchain/chat_models/openai";
import { PromptTemplate } from "langchain/prompts";
import dotenv from 'dotenv';

dotenv.config();

const openAIApiKey = process.env.OPENAI_API_KEY;
const llm = new ChatOpenAI({ openAIApiKey });

const standaloneQuestionTemplate = 'Given a question, convert it to a standalone question. Question: {questionText}, standalone question:';
const standaloneQuestionPrompt = PromptTemplate.fromTemplate(standaloneQuestionTemplate);
const standaloneQuestionChain = standaloneQuestionPrompt.pipe(llm);

const response = await standaloneQuestionChain.invoke({ questionText: "I'm really hungry, what's on the menu? I want to eat right away, whatever's cheapest!" });

console.log(response.content);