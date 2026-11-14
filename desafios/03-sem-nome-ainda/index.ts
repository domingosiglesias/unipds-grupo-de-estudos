import 'dotenv/config';
import { ChatOpenRouter } from '@langchain/openrouter';
import { END, MessagesValue, START, StateGraph, StateSchema, Send } from "@langchain/langgraph";
import { AIMessage, createAgent, HumanMessage } from 'langchain';
import { getMoreContextFromDatabase } from './tools.js';
import { z } from 'zod';
import { tool } from "langchain"



const llm = new ChatOpenRouter({ 
    apiKey: process.env.OPENROUTER_API_KEY,
    model: process.env.OPENROUTER_MODEL,
});


export const States = new StateSchema({
    messages: MessagesValue,
    next: z.enum(["agent_support", "agent_memory"]),
});

export type States = typeof States.State;

const AgentSupport = createAgent({
    model: llm,
    systemPrompt: `
    REGRA: Responda sempre em português do Brasil.
    INSTRUÇÃO:
    Você é um assistente pessoal de TI de um grupo de estudos da UNIPDS.
    `.trim(),
    tools: [ ],
});

const AgentWithDatabase = createAgent({
    model: llm,
    systemPrompt: `
    REGRA: Responda sempre em português do Brasil.
    INSTRUÇÃO:
    Você é um assistente pessoal de TI de um grupo de estudos da UNIPDS.
    Se o contexto não tiver a resposta, diga que não encontrou na base.
    E responda com sempre com uma pergunta.
    `.trim(),
    tools: [ 
        getMoreContextFromDatabase
    ],
});



export const subAgentSupportTool = tool(
    async ({ query }: { query: string }) => {
        const response = await AgentSupport.invoke({ messages: [ new HumanMessage(query) ] });
        return response.messages.at(-1)?.content as string;
    },
    {
        name: "subAgentSupport",
        description: `Use essa tools para responder perguntas sobre a UNIPDS`,
        schema: z.object({
            query: z.string().describe("A pergunta sobre a UNIPDS"),
        }),
    }
)

export const subAgentGetMoreContextFromDatabaseTool = tool(
    async ({ query }: { query: string }) => {
        const response = await AgentWithDatabase.invoke({ messages: [ new HumanMessage(query) ] });
        return response.messages.at(-1)?.content as string;
    },
    {
        name: "subAgentGetMoreContextFromDatabase",
        description: `Use essa tools para buscar informações no banco de dados`,
        schema: z.object({
            query: z.string().describe("A consulta para buscar informações no banco de dados"),
        }),
    }
)



const AgentOrchestratorResponseSchema = z.object({
    next: z.enum(["agent_support", "agent_memory"]),
});

const AgentOrchestrator = createAgent({
    model: llm,
    systemPrompt: `
    REGRA: Responda sempre em português do Brasil.
    INSTRUÇÃO:
    Você é um orquestrador de agentes.
    Você deve decidir qual agente usar com base na mensagem do usuário.
    Se o usuário perguntar sobre a UNIPDS, use o agente subAgentSupportTool.
    Se o usuário perguntar sobre o banco de dados, use o agente subAgentGetMoreContextFromDatabaseTool.
    `.trim(),
    tools: [ 
        subAgentSupportTool,
        subAgentGetMoreContextFromDatabaseTool,
    ],
    // responseFormat: AgentOrchestratorResponseSchema,
});


export function createGraph() { 


    const workflow = new StateGraph(States)
    // .addNode('agent_support', async (state: States) => { 
    //     const response = await AgentSupport.invoke( { messages: state.messages } );
    //     const lastMessage = response.messages.at(-1)?.content as string;
    //     return {
    //         messages: [ 
    //             new AIMessage(lastMessage) 
    //         ]
    //     }
    // })
    // .addNode('agent_memory', async (state: States) => { 
    //     const response = await AgentWithDatabase.invoke( { messages: state.messages } );
    //     const lastMessage = response.messages.at(-1)?.content as string;
    //     return {
    //         messages: [ new AIMessage(lastMessage) ]
    //     }
    // })
    .addNode('agent_orchestrator', async (state: States) => { 
        const response = await AgentOrchestrator.invoke( { messages: state.messages } );
        return {
            next: response.structuredResponse.next
        }
    })
    .addEdge(START, 'agent_orchestrator')
    // .addConditionalEdges('agent_orchestrator', (state) => state.next, { 
    //     'agent_support': 'agent_support',
    //     'agent_memory': 'agent_memory'
    // }) 
    // .addEdge('agent_memory', END)
    .addEdge('agent_orchestrator', END)
    ;


    return workflow.compile();

}


export const graph = createGraph();