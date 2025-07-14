// a TSV parser that parses the data incrementally in chunks
const tsvChunkedParser = () => {
  const textDecoder = new TextDecoder("utf-8");
  let columnHeadings;
  let previousChunk = "";

  return {
    parseChunk(chunk) {
      // decode buffer to string
      const chunkStr =  textDecoder.decode(chunk);

      // connect the previous chunk with the new chunk
      const textData =  previousChunk + chunkStr;

      // get new lines based on the new chunk and previous chunk
      const lines = textData.split("\n");

      // the first line is our column headings
      if (!columnHeadings) {
        columnHeadings = lines[0].trim().split("\t");
        lines.shift();
      }
      // the last line is probably partial - so append to the next chunk
      previousChunk = lines.pop();

      // convert each row to an object
      const items = lines
        .map(row => {
          // skip empty rows or rows that are just whitespace
          let _row = row.trim();
          if (_row === "") {
            return null;
          }
          // the row may contain a trailing \t or empty string
          // do not trim the row, as it may contain meaningful whitespace
          // const cells = row.trim().split("\t");
          const cells = row.split("\t");

          // never skip rows, even if they are incomplete
          // if (cells.length !== columnHeadings.length) {
          //   return null;
          // }

          // most of the time, the number of cells will match the number of column headings
          // but it is possible that the row is incomplete
          let rowValue = {};
          columnHeadings.forEach((h, i) => {
            // the cell number may be less than the number of column headings
            // in which case, we will put null in the rowValue
            if (i < cells.length) {
              rowValue[h] = cells[i];
            } else {
              rowValue[h] = null;
            }
          });
          return rowValue;
        })
        .filter(i => i);

      return items;
    }
  };
};

/**
 * The worker receives a message with the URL of the TSV file to be parsed.
 * 
 * {
 *   url: string
 * }
 * 
 * Just calling the worker with the URL will start the process:
 * 
 * worker_web_streaming_tsv.postMessage({ url: "https://example.com/data.tsv" });
 */
onmessage = async ({ data: msg }) => {
  let totalBytes = 0;

  const tsvParser = tsvChunkedParser();
  const response = await fetch(msg.url, {
    method: 'GET',
    headers: {
      'Accept-Encoding': 'gzip',
    },
  });

  if (!response.body) {
    throw Error("ReadableStream not yet supported in this browser.");
  }

  const streamedResponse = new Response(
    new ReadableStream({
      start(controller) {
        const reader = response.body.getReader();

        const read = async () => {
          const { done, value } = await reader.read();
          if (done) {
            controller.close();
            postMessage({ items: [], totalBytes: totalBytes, finished: true});
            return;
          }

          const items = tsvParser.parseChunk(value);
          totalBytes += value.byteLength;
          postMessage({ items, totalBytes, finished: false });

          controller.enqueue(value);
          read();
        };

        read();
      }
    })
  );
};