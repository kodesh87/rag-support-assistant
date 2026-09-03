import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { ChatOpenAI } from 'langchain/chat_models/openai'
import { ChatPromptTemplate } from "langchain/prompts";
import { ChatMessageHistory } from "langchain/memory";
import { StringOutputParser } from 'langchain/schema/output_parser'
import { retriever } from './utils/retriever.js'
import { RunnablePassthrough, RunnableSequence } from "langchain/schema/runnable"
import bodyParser from 'body-parser';
import { createRetrieverTool, OpenAIAgentTokenBufferMemory } from "langchain/agents/toolkits";
import { initializeAgentExecutorWithOptions } from 'langchain/agents';

const openAIApiKey = process.env.OPENAI_API_KEY
const llm = new ChatOpenAI({ modelName: 'gpt-3.5-turbo-1106', openAIApiKey });

const chatHistory = new ChatMessageHistory();

const chatPromptMemory = new OpenAIAgentTokenBufferMemory({
  llm: new ChatOpenAI({ modelName: 'gpt-3.5-turbo-1106', openAIApiKey }),
  memoryKey: "chat_history",
  outputKey: "output",
  chatHistory,
});


// standalone prompt template
const standaloneQuestionTemplate = 'Given a question, convert it to a standalone question. question: {question} standalone question:'
const standaloneQuestionPrompt = ChatPromptTemplate.fromTemplate(standaloneQuestionTemplate)

const AGENT_PERSONA = `You are a helpful and enthusiastic support bot who can answer a given question about Nimbus Cloud based on the context provided. Try to find the answer in the context. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@nimbuscloud.dev. Don't try to make up an answer, don't talk about other VPS providers. Always speak as if you are a Nimbus Cloud agent chatting to a friend, and reply in the same language as the question.`

// retrieval tool available to the agent
const tool = createRetrieverTool(retriever, {
    name: "faq_nimbus_cloud",
    description:
      "Frequently Asked Questions about Nimbus Cloud.",
  });

// chains
const standaloneQuestionChain = RunnableSequence.from([
    standaloneQuestionPrompt,
    llm,
    new StringOutputParser()
]);
const answerChain = await initializeAgentExecutorWithOptions(
    [tool],
    llm,
    {
        agentType: "openai-functions",
        memory: chatPromptMemory,
        returnIntermediateSteps: true,
        agentArgs: {
          prefix: AGENT_PERSONA,
        },
    }
);

// main chain: rewrite the question standalone, then let the agent decide
// whether/how to call the retriever tool before answering
const chain = RunnableSequence.from([
    {
        standalone_question: standaloneQuestionChain,
        original_input: new RunnablePassthrough()
    },
    {
        input: ({ original_input }) => original_input.question
    },
    answerChain
]);

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.static(path.join(__dirname, 'public')));
app.use(bodyParser.json());
app.use(express.urlencoded({ extended: true }));

app.post('/api/chat', async (req, res) => {
  const requestData = req.body;

  const response = await chain.invoke({
      question: requestData.question
  })

  res.status(200).json({ message: 'Data received successfully', data: { question: requestData.question, answer: response.output } });
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
});