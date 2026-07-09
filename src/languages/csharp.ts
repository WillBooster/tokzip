import { csharpModule } from '../generated/csharp.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(csharpModule);

export { csharpModule };
