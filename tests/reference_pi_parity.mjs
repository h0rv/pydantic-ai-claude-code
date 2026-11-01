import assert from 'node:assert/strict';
import {readFileSync, writeFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {resolve, join} from 'node:path';
import {pathToFileURL} from 'node:url';

// Run with Bun. This generator only calls pure upstream exports.
// Usage: bun reference_pi_parity.mjs UPSTREAM_CHECKOUT OUTPUT_JSON
const args = process.argv.slice(2);
assert.equal(args.length, 2, 'Usage: bun reference_pi_parity.mjs UPSTREAM_CHECKOUT OUTPUT_JSON');
const upstream = resolve(args[0]);
const outputPath = resolve(args[1]);
const sourceCommit = execFileSync('git', ['-C', upstream, 'rev-parse', 'HEAD'], {encoding:'utf8'}).trim();
assert.equal(sourceCommit, '70abb65186592ca788b29b3db22c3160d629c41f');
const {buildExtraBody, prepareFrames} = await import(pathToFileURL(join(upstream, 'src/transcript.ts')).href);
const nativeTools = [{name:'mcp__pi__lookup',description:'Lookup',input_schema:{type:'object',properties:{key:{type:'string'}}}}];
const extraCases = [
 {name:'default',model:'claude-sonnet-4-5',tools:nativeTools},
 {name:'max-tokens',model:'claude-sonnet-4-5',tools:nativeTools,options:{maxTokens:64}},
 {name:'adaptive-high',model:'claude-sonnet-4-6',tools:nativeTools,options:{maxTokens:64,reasoning:'high'}},
 {name:'minimal',model:'claude-sonnet-4-6',tools:[],options:{reasoning:'minimal'}},
].map(input=>({input,output:buildExtraBody(input.model,input.tools,input.options)}));
const messages = [
 {role:'system',content:'Application instruction',timestamp:0},
 {role:'user',content:'Inspect key',timestamp:1},
 {role:'assistant',content:[{type:'thinking',thinking:'Synthetic reasoning',thinkingSignature:'synthetic-signature'},{type:'text',text:'Checking'},{type:'toolCall',id:'call_1',name:'lookup',arguments:{key:'A'}}],timestamp:2},
 {role:'toolResult',toolCallId:'call_1',toolName:'lookup',content:[{type:'text',text:'Value A'}],isError:false,timestamp:3},
 {role:'user',content:'Summarize',timestamp:4},
];
const frames=prepareFrames(messages);
assert.equal(frames.length,3);
assert.equal(frames[0].shouldQuery,false);
assert.equal(frames[1].message.content[2].name,'mcp__pi__lookup');
assert.equal(frames[2].message.content.length,2);
const peerPackage=JSON.parse(readFileSync(join(upstream,'node_modules/@earendil-works/pi-ai/package.json'),'utf8'));
writeFileSync(outputPath,JSON.stringify({sourceCommit,peer:{name:peerPackage.name,version:peerPackage.version},buildExtraBody:extraCases,prepareFrames:{input:messages,output:frames}},null,2)+'\n');
console.log(JSON.stringify({sourceCommit, peerVersion:peerPackage.version, extraCases:extraCases.length, frameRoles:frames.map(f=>f.type)},null,2));
