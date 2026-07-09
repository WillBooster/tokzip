import { textModule } from '../generated/text.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(textModule);

export { textModule };
