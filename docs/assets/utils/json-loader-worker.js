self.onmessage=async({data:e})=>{try{const a=await(await fetch(e)).json();self.postMessage(a)}catch(s){self.postMessage({error:s.message})}};
